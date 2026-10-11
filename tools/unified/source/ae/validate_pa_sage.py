#!/usr/bin/env python3
"""Validate frozen PA/SAGE reconstruction inputs or small GPU semantics, not paper performance."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ae.common import read, sha, require, verify_release, write
from ae.freeze import verify


def input_audit(cfg):
    import numpy as np
    current = read(ROOT / 'configs/papers/protocol.json')
    data = current['dataset']
    split_ids, splits = {}, {}
    for name, desc in data['splits'].items():
        require(sha(desc['path']) == desc['sha256'], 'Split checksum: ' + name)
        with gzip.open(desc['path'], 'rt') as f:
            ids = np.fromiter((int(line) for line in f if line.strip()), dtype=np.int64)
        require(len(ids) == desc['count'], 'Split count: ' + name)
        require(len(np.unique(ids)) == len(ids), 'Duplicate split IDs: ' + name)
        require(ids.min() >= 0 and ids.max() < data['num_nodes'], 'Split bounds: ' + name)
        split_ids[name] = ids
        splits[name] = {'count': len(ids), 'sha256': desc['sha256']}
    all_ids = np.concatenate(list(split_ids.values()))
    require(len(np.unique(all_ids)) == len(all_ids), 'Train/valid/test overlap')
    labels_desc = data['label_identity']
    require(sha(labels_desc['path']) == labels_desc['sha256'], 'Labels checksum')
    labels = np.load(labels_desc['path'], mmap_mode='r').reshape(-1)
    for name, ids in split_ids.items():
        values = labels[ids]
        require(np.isfinite(values).all() and (values == np.floor(values)).all(), 'Invalid labels')
        require(values.min() >= 0 and values.max() < cfg['reconstruction_choices']['num_classes'], 'Label bounds')
        splits[name]['label_range'] = [int(values.min()), int(values.max())]
    features = np.load(data['source_features']['path'], mmap_mode='r')
    require(list(features.shape) == [data['num_nodes'], 128] and features.dtype == np.float32, 'Feature geometry')
    probes = np.unique(np.linspace(0, len(features) - 1, 1024, dtype=np.int64))
    require(np.isfinite(features[probes]).all(), 'Feature probe contains nonfinite values')
    bundle = read(ROOT / 'data/papers/bundle/manifest.json')
    trace = read(ROOT / 'data/papers/valid_trace/manifest.json')
    import digit_paths
    test_trace = read(Path(digit_paths.locations()['papers_test_trace']) / 'manifest.json')
    target = cfg['paper_specified']
    choices = cfg['reconstruction_choices']
    diffs = []
    def compare(field, actual, expected):
        if actual != expected:
            diffs.append({'field': field, 'current': actual, 'required': expected})
    compare('protocol.training.fanouts', current['training']['fanouts_forward_layer_order'], target['fanouts'])
    compare('bundle.group_size', bundle['grouping']['group_size'], target['group_size'])
    compare('protocol.bfs.enabled', current['digit_full']['bfs']['enabled_in_main_comparison'], True)
    compare('valid_trace.fanouts', trace['sampling']['fanouts'], target['fanouts'])
    compare('test_trace.fanouts', test_trace['sampling']['fanouts'], target['fanouts'])
    # Inspect executable literals as well as JSON: the legacy runner does not read these settings from JSON.
    runner = ast.parse((ROOT / 'ae/papers/runner.py').read_text())
    for node in ast.walk(runner):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
        if name == 'GIDS':
            for kw in node.keywords:
                if kw.arg == 'cache_size':
                    compare('runner.py:%d.cache_size_MiB' % node.lineno, ast.literal_eval(kw.value), choices['gpu_cache_bytes'] // 2**20)
        if name in ('DiGiTNeighborSampler', 'NeighborSampler'):
            compare('runner.py:%d.fanouts' % node.lineno, ast.literal_eval(node.args[0]), target['fanouts'])
    blockers = ['The full PA training adapter for this reconstruction protocol is not implemented.',
                'Matching g2 layout, CPU-hot profiling, BFS order, evaluation traces and SSD readback receipts are not prepared.',
                'GPU/host admission must be recomputed for the rebuilt data and 4 GiB cache.',
                'Native binaries have not been rebuilt from the frozen sources on a clean environment.']
    return {'passed': True, 'scope': 'official_PA_input_identity_and_existing_runner_alignment_audit',
            'splits': splits, 'split_disjoint': True, 'labels_sha256_checked': True,
            'feature_shape': list(features.shape), 'feature_finite_probe_rows': len(probes),
            'full_feature_hash_checked': False, 'full_graph_hash_checked': False,
            'configuration_mismatches': diffs, 'blockers': blockers,
            'paper_training_ready': False, 'training_run': False, 'raw_ssd_io': False}


def small_gpu_check(cfg):
    import numpy as np
    import torch
    import dgl
    import digit
    import digit.sampler as sampling
    from digit.reorganization import reorganize_to_bundle
    from models import SAGE
    require(torch.cuda.is_available(), 'A visible idle CUDA device is required')
    require(str(ROOT / 'ae/papers/runtime') in sampling.__file__, 'Wrong sampler import')
    torch.set_num_threads(1)
    dgl.utils.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Existing invariant tests run against the extracted runtime, including CUDA cases.
    digit.__path__.append(str(ROOT / 'evaluation/digit'))
    modules = ['digit.tests.test_artifacts', 'digit.tests.test_reorganization',
               'digit.tests.test_bfs_order', 'digit.tests.test_sampler']
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    tested = unittest.TextTestRunner(verbosity=1).run(suite)
    require(tested.wasSuccessful() and not tested.skipped, 'Correctness tests failed or skipped')
    n, dim, batch, steps = 128, 128, 16, 4
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, dim)).astype(np.float32)
    labels = rng.integers(0, 172, n, dtype=np.int64)
    indices = np.concatenate([(np.arange(25, dtype=np.int64) + v) % n for v in range(n)])
    indptr = np.arange(n + 1, dtype=np.int64) * 25
    destinations = np.repeat(np.arange(n, dtype=np.int64), 25)
    graph = dgl.graph((torch.from_numpy(indices), torch.from_numpy(destinations)), num_nodes=n)
    graph.create_formats_()
    graph.pin_memory_()
    target, choices = cfg['paper_specified'], cfg['reconstruction_choices']
    arms = {}
    initial_hashes = []
    inner_edges = {}
    grouped = 0
    with tempfile.TemporaryDirectory(prefix='digit-pa-sage-') as temporary:
        bundle = reorganize_to_bundle(indptr, indices, features, temporary,
                    dataset_name='synthetic', dataset_size='PA-128D-semantics',
                    group_size=target['group_size'], replication_ratio=target['replication_ratio'],
                    page_size=choices['io_page_bytes'], minimum_transfer_bytes=choices['io_page_bytes'],
                    hot_nodes=np.arange(8, dtype=np.int64), seed=0)
        reordered = np.load(Path(temporary) / 'reordered_features.npy', mmap_mode='r')
        for arm in ('gids', 'digit_full'):
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            dgl.seed(0)
            model = SAGE(dim, target['hidden_dim'], choices['num_classes'],
                         num_layers=target['layers'], dropout=choices['dropout']).cuda()
            h = hashlib.sha256()
            for name, value in model.state_dict().items():
                h.update(name.encode()); h.update(value.cpu().numpy().tobytes())
            initial_hashes.append(h.hexdigest())
            optimizer = torch.optim.Adam(model.parameters(), **choices['optimizer']['kwargs'])
            sampler = (sampling.DiGiTNeighborSampler(target['fanouts'], bundle,
                        cuda_mode='required', metadata_mode='gpu_i32', random_seed=0)
                       if arm == 'digit_full' else dgl.dataloading.NeighborSampler(target['fanouts'], replace=False))
            losses = []
            for step in range(steps):
                torch.manual_seed(step); torch.cuda.manual_seed_all(step); dgl.seed(step)
                roots = torch.arange(step * batch, (step + 1) * batch, device='cuda')
                inp, out, blocks = sampler.sample_blocks(graph, roots)
                require(torch.equal(out, roots), 'Output node order changed')
                for layer, block in enumerate(blocks):
                    u, v = block.edges(order='eid')
                    eu, ev = graph.find_edges(block.edata[dgl.EID].cpu())
                    require(torch.equal(block.srcdata[dgl.NID][u].cpu(), eu), 'Edge source mismatch')
                    require(torch.equal(block.dstdata[dgl.NID][v].cpu(), ev), 'Edge destination mismatch')
                    if layer > 0:
                        eids = torch.sort(block.edata[dgl.EID].cpu()).values
                        if arm == 'gids': inner_edges[step, layer] = eids
                        else: require(torch.equal(eids, inner_edges[step, layer]), 'Inner-layer sampling changed')
                logical = inp.cpu().numpy()
                if arm == 'digit_full':
                    rows = sampling.feature_rows_from_blocks(blocks).cpu().numpy()
                    actual = np.asarray(reordered[rows])
                    require(np.array_equal(actual.view(np.uint32), features[logical].view(np.uint32)), 'Physical feature mapping mismatch')
                    grouped += int(blocks[0].srcdata[sampling.DIGIT_STORAGE_IS_GROUP].sum())
                    require(bool(torch.all(blocks[0].in_degrees() <= target['fanouts'][0])), 'Outer fanout budget exceeded')
                else: actual = features[logical]
                x = torch.from_numpy(actual.copy()).cuda()
                y = torch.from_numpy(labels[out.cpu().numpy()].copy()).cuda()
                pred = model(blocks, x)
                loss = torch.nn.functional.cross_entropy(pred, y)
                require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite gradients')
                optimizer.step(); losses.append(float(loss.detach()))
            arms[arm] = {'optimizer_updates': steps, 'losses': losses, 'initial_model_sha256': initial_hashes[-1]}
        require(initial_hashes[0] == initial_hashes[1], 'Unpaired model initialization')
        require(grouped > 0, 'Grouped path was never exercised')
        geometry = bundle.io_geometry.to_dict()
    return {'passed': True, 'scope': 'synthetic_sampler_feature_mapping_and_SAGE_optimizer',
            'existing_correctness_tests': tested.testsRun, 'skipped_tests': len(tested.skipped),
            'synthetic_nodes': n, 'synthetic_batch_size': batch, 'fanouts': target['fanouts'],
            'feature_dim': dim, 'group_size': target['group_size'], 'geometry': geometry,
            'grouped_feature_rows_exercised': grouped, 'inner_sampling_parity': True,
            'arms': arms, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'dgl': dgl.__version__, 'sampler_module': str(Path(sampling.__file__).relative_to(ROOT)),
            'raw_ssd_io': False, 'GIDS_feature_loader_exercised': False,
            'GPU_feature_cache_exercised': False, 'PA_dataset_training_run': False,
            'paper_training_ready': False, 'performance_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inputs', 'tiny'])
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output must be fresh')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['DGLBACKEND'] = 'pytorch'
    os.environ['USE_DETERMINISTIC_ALG'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(ROOT / 'ae/papers/runtime'))
    release = verify_release()
    snapshot = verify(args.snapshot, ROOT)
    cfg = read(ROOT / 'configs/paper/pa_sage.json')
    try:
        result = input_audit(cfg) if args.mode == 'inputs' else small_gpu_check(cfg)
        require(verify_release() == release, 'Sources changed during validation')
    except Exception as error:
        write(args.output, {'passed': False, 'mode': args.mode, 'error': str(error),
                           'snapshot': snapshot, 'release_sha256': release})
        raise
    result.update(snapshot=snapshot, release_sha256=release,
                  protocol_sha256=sha(ROOT / 'configs/paper/pa_sage.json'))
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
