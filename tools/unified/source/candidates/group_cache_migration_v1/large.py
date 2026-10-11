"""Full profiling inside existing anonymous-memory, NUMA-checked lifecycles."""
import hashlib
import json
import time
from itertools import islice
from pathlib import Path
import numpy as np
from .common import require, write
from .core import profile, select_hot, audit, batches, validate_order, digest_array
from .io import read_array, selected_scores, save_array


def graph_files(accepted):
    return accepted['stages']['graph']['files']


def root_digest(receipt):
    return receipt['sha256']


def prepare_cache(arm, native, sampling, accepted, previous_hot, output, check, event):
    files = graph_files(accepted)
    matched = files['digit_roots.i64']
    metadata = dict(matched_bfs_roots_sha256=matched['sha256'],
                    capacity_rows=accepted['cpu_cache_rows'], gpu_cache_bytes=4*2**30,
                    gpu_reset='fresh provider after profiling; preload must leave residency zero',
                    profile=None, audit=None, hot_policy='RevPR' if arm == 'gids' else 'full_epoch_Freq_RevPR_fallback')
    if arm == 'gids': return None, metadata
    started = time.perf_counter()
    # Maximum live extra: counts 4N + flags N + train/order ~2.4N plus selection.
    # This stage runs BEFORE the >=40 GiB CPU feature cache and N*4 slot lookup.
    canonical = read_array(files['bfs.i64'], check=check)
    train = read_array(files['train.i64'], check=check)
    validate_order(canonical, train, native.graph.nodes); del train
    window = np.concatenate(list(islice(batches(canonical),320)))
    require(digest_array(window) == matched['sha256'], 'Profile roots differ from formal BFS batches')
    del window
    _, Sampler = sampling.select('digit')
    def factory(seed):
        sampler = Sampler(native, grouped=True, seed=seed)
        def sample(roots, index):
            inp, _, blocks = sampler.sample_blocks(roots, index)
            ids = inp.detach().cpu().numpy().astype(np.int64, copy=True) if hasattr(inp,'detach') else np.asarray(inp,dtype=np.int64).copy()
            del blocks
            return ids
        return sample
    counts, receipt = profile(canonical, native.graph.nodes, factory, event, check)
    write(output/'full_epoch_profile.json', receipt)
    count_file = save_array(output/'frequency_counts.u32', counts, check)
    baseline = read_array(files['gids_hot.i64'], check=check)
    scores = score_receipt = None
    if receipt['observed_nodes'] < len(baseline):
        if accepted.get('dataset') == 'CL':
            from candidates.cl_night_prepare_v9 import protocol as cp
            rank = Path(cp.metadata()['revpr']['path']).with_name('revpr.f64')
            evidence = Path('/home/embed/digit/results/cl_prepare_guarded_20261003_v7')/rank.parent.parent.name/'worker.json'
            record = json.loads(evidence.read_text())['files']['revpr.f64']
            require(record['path'] == str(rank), 'CL rank binding mismatch')
            expected = record['sha256']
        else:
            from candidates.ukl_runtime_prepare_v11r1.protocol import REVPR_PATH
            rank = REVPR_PATH.with_name('revpr.f64')
            record = json.loads(rank.with_name('binding.json').read_text())
            require(record['files']['hot_nodes.i64']['sha256'] == files['gids_hot.i64']['sha256'], 'UKL rank/hot mismatch')
            expected = record['files']['revpr.f64']['sha256']
        scores, score_receipt = selected_scores(rank, native.graph.nodes, baseline, check, expected)
    hot, selection = select_hot(counts, len(baseline), baseline, scores, check)
    del scores, counts
    hot_file = save_array(output/'full_epoch_hot.i64', hot, check)
    write(output/'hot_selection.json', dict(selection, file=hot_file, counts=count_file, revpr_scores=score_receipt))
    eligibility = dict(complete=False, reason='This study measures native cache serving after one profiling epoch; no extra audit epochs')
    del baseline, canonical
    import torch
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    # Sampling may consume RNG state, but the formal model/optimizer must start
    # from the same seed as the GIDS arm. Formal sampler is constructed later.
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    metadata.update(profile=receipt, selection=selection, audit=eligibility,
                    hot_file=hot_file, counts_file=count_file,
                    preparation_seconds=time.perf_counter()-started,
                    sampler_seed_reset=0, native_gpu_feature_cache_not_yet_created=True)
    return hot, metadata


def validate_receipt(report, arm):
    try:
        value = report['cache_epoch']
        require(value['capacity_rows'] == report['cache']['cpu_rows'], 'Capacity mismatch')
        require(value['gpu_cache_bytes'] == report['cache']['gpu_feature_bytes'], 'GPU capacity mismatch')
        require(report['cpu_affinity']['cpus'] == [2] and report['cpu_affinity_initial']['cpus'] == [2], 'Affinity mismatch')
        require(report['counters']['whole']['raw_allocator_cache']['resident_pages_start'] == 0, 'GPU cache not cold')
        for name in ('whole','measured'):
            group=report['group_sampling'][name]
            require(0<=group['grouped_sampled_edges']<=group['outer_sampled_edges'],'Invalid sampled grouping counts')
            require(group['grouped_sampled_edges']%2==0,'Partial group emitted')
            if arm=='gids':require(group['grouped_sampled_edges']==0,'GIDS used grouped sampling')
        if arm == 'digit':
            require(value['profile']['complete'] and value['profile']['epochs'] == 1, 'Incomplete profile')
            require(value['profile']['seed'] == 23 and value['profile']['feature_reads'] == 0, 'Invalid profiling')
            require(value['hot_file']['sha256'] == value['selection']['hot_sha256'], 'Hot-set provenance')
            require(value['native_gpu_feature_cache_not_yet_created'], 'Profiling warmed GPU feature cache')
            require(report['numa_admission']['adapter_sha256'] == report['manifest_sha256'], 'NUMA evidence changed')
        return True
    except (KeyError,TypeError,ValueError,RuntimeError,OSError): return False


def summarize_native(series):
    pair = series['pairs'][0]; reports = {}
    for arm in ('gids','digit'):
        acceptance = json.loads(Path(pair[arm]['path']).read_text())
        require(acceptance['passed'], 'Unaccepted worker')
        reports[arm] = acceptance['worker']
    a,b = reports['gids'], reports['digit']
    require(a['cache_epoch']['matched_bfs_roots_sha256'] == b['cache_epoch']['matched_bfs_roots_sha256'], 'Mismatched roots')
    require(a['cache']['cpu_feature_bytes'] == b['cache']['cpu_feature_bytes'], 'Unequal CPU bytes')
    require(a['initial_model_sha256']==b['initial_model_sha256'], 'Unmatched initial model')
    return dict(passed=True, scope='one native paired 20+300-batch window after one complete profiling epoch',
                speedup=a['seconds']/b['seconds'], aggregation='one pair, no max selection',
                arms={arm:dict(seconds=r['seconds'], serving=r['counters']['measured']['serving'],
                    cache=r['cache'], cache_epoch=r['cache_epoch'], group_sampling=r['group_sampling'], group_reuse=r['group_reuse'], ssd=r['counters']['measured']) for arm,r in reports.items()},
                accuracy_evaluated=False, full_epoch_training=False)
