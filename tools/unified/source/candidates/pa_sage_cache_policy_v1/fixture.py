"""Bounded DGL g2/r20 fixture; no production arrays, GPU contexts or SSD access."""
import copy
import numpy as np
from .common import ARMS, read, write_new, digest, require
from .selection import degree_scores, reverse_pagerank, topk
from .protocol import compile_protocol, arms
from .oracle import FeatureOracle
from .training import profile_cpu, cpu_epoch


def exercise(output):
    from candidates.pa_sage_layout_shared_resume_v3.common import setup
    setup()
    import torch
    import dgl
    from candidates.pa_sage_layout_sweep_v1.tests import fixture
    from candidates.pa_sage_layout_shared_resume_v3.build import Context, build_layout
    from candidates.pa_sage_layout_shared_resume_v3.overlay import build_overlay
    from candidates.pa_sage_layout_shared_resume_v3.bundle import load_metadata
    from candidates.pa_sage_bidir_native_v2.overlay import apply
    from digit.sampler import DiGiTNeighborSampler
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dgl.utils.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU fixture must not start CUDA')
    output.mkdir(parents=True, exist_ok=False)
    n, k = 96, 8
    inputs, edges, bidir, reverse = fixture(output / 'inputs', n=n, hot_count=k)
    base = output / 'layout'
    ctx = Context(dict(id='g2_r20', group_size=2, replica_percent=20), inputs, n, edges, k, base, fixture=True)
    build_layout(ctx)
    build_overlay(base, inputs['indptr']['path'], bidir, reverse, base / 'overlay', fixture=True)
    bundle = load_metadata(base)
    ptr, idx = (np.load(bidir / ('original_' + key + '.npy')) for key in ('indptr', 'indices'))
    graph = dgl.graph(('csc', (torch.from_numpy(ptr), torch.from_numpy(idx), torch.empty(0, dtype=torch.int64))), num_nodes=n)
    bundle = apply(bundle, np.load(base / 'overlay/reordered_indptr.npy'),
                   np.load(base / 'overlay/reordered_indices.npy'), len(idx))
    p = compile_protocol()
    p = copy.deepcopy(p)
    p['batch_size'] = 8
    p['arms'] = arms(k, 16 * 4096)
    train = np.arange(n, dtype=np.int64)
    labels = train % p['classes']
    features = np.load(inputs['features']['path'])
    storage = bundle.arrays['storage_to_node']
    payload = np.zeros((len(storage), 128), dtype=np.float32)
    valid = storage >= 0
    payload[valid] = features[storage[valid]]

    def sampler(seed):
        return DiGiTNeighborSampler(p['fanouts'], bundle, cuda_mode='disabled',
                                   metadata_mode='cpu_eid', random_seed=seed)

    freq, profile = profile_cpu(sampler(23), graph, train, n, 23, batches=3, batch_size=p['batch_size'])
    scores = dict(degree=degree_scores(ptr, idx), revpr=reverse_pagerank(ptr, idx), freq=freq)
    hots = {name: topk(values, k) for name, values in scores.items()}
    reports = {}
    for arm in ARMS:
        policy = p['arms'][arm]
        hot = hots[policy['selection']]
        oracle = FeatureOracle(storage, payload, features, hot, policy['gpu_feature_cache_bytes'])
        report = cpu_epoch(p, sampler(p['seed']), graph, train, labels, oracle)
        report.update(arm=arm, hot_nodes=hot.tolist(), hot_sha256=digest(hot), policy=policy)
        reports[arm] = report
        write_new(output / (arm + '.json'), report)
    baseline = reports['degree']
    for arm, report in reports.items():
        for key in ('initial_parameters_sha256', 'final_parameters_sha256', 'sampling_trace_sha256', 'root_sha256', 'losses', 'shapes'):
            require(report[key] == baseline[key], 'Cache policy changed training semantics: ' + key)
        require(report['metrics']['storage_request_sha256'] == baseline['metrics']['storage_request_sha256'], 'Cache changed physical request sequence')
    require(reports['freq']['hot_nodes'] == reports['digit']['hot_nodes'], 'Freq/DiGiT hot sets differ')
    require(not torch.cuda.is_initialized(), 'CPU fixture opened CUDA')
    result = dict(passed=True, nodes=n, graph_edges=len(idx), updates_per_arm=12,
        arms=list(reports), independent_profile=profile,
        identical_sampling_features_losses_and_final_weights=True,
        cuda_initialized=False, raw_ssd_access=False, native_execution=False,
        cpu_feature_bytes=k * 512, digit_gpu_feature_cache_bytes=16 * 4096,
        reports={arm: str(output / (arm + '.json')) for arm in ARMS},
        scope='Tiny complete CPU epochs only. Oracle hit ratios must not be used as paper measurements.')
    write_new(output / 'summary.json', result)
    return result
