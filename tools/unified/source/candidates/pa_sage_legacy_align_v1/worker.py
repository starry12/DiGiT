"""A fresh native process, either tail-sensitive smoke or full official epoch."""
import argparse
import fcntl
import math
import random
import resource
from .common import *


def snapshot(loader):
    return dict(feature=loader.get_feature_access_stats(), gpu=loader.get_gpu_cache_stats(),
                device=loader.get_device_io_stats())


def run(folder, smoke):
    p = read(HERE / 'protocol.json')
    execution = verify()
    binding = read(OUT / 'inputs.json')
    check_inputs(binding)
    require(os.geteuid() == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '2',
            'This native candidate requires administrator execution on physical GPU 2')
    from ae.common import check_device, check_payload, host
    require(host() >= p['host_admission_gib'] * 2**30, 'Host memory admission failed')
    check_device()
    require(check_payload('papers_gids') == binding['ssd_payload'], 'SSD payload receipt changed')
    setup()
    import numpy as np
    import torch
    import dgl
    import BAM_Feature_Store
    from GIDS import GIDS_DGLDataLoader, get_sample_record, get_fetch_feature_record
    from .loader import LegacyAlignedGIDS
    from .graph import load_graph, verify_blocks
    from models import SAGE
    from ae.pa_sage.common import splits
    require(Path(__import__('sys').modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve() == BINARY,
            'Wrong native binary imported')
    require(torch.cuda.mem_get_info()[0] >= p['gpu_min_free_gib'] * 2**30, 'GPU memory admission failed')
    folder.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    progress(folder, 'loading_directed_graph', smoke=smoke)
    graph, arrays = load_graph(binding, p['nodes'], p['edges'])
    progress(folder, 'initializing_512b_cache')
    payload = binding['ssd_payload']
    loader = LegacyAlignedGIDS(page_size=512, off=payload['offset'], cache_dim=128,
        num_ele=p['nodes'] * 128, num_ssd=1, ssd_list=[0], cache_size=4096, ctrl_idx=0,
        window_buffer=True, wb_size=2, accumulator_flag=True, feature_index_mode='logical',
        gpu_cache_policy='legacy', cpu_feature_path='mapped', mixed_io=False, device_io_stats=True)
    loader.set_required_storage_access(**p['accumulator_parameters'])
    require(math.isclose(loader.required_accesses, p['expected_required_accesses']), 'Accumulator threshold changed')
    hot = torch.load(p['pin_file'], map_location='cpu')[:p['cpu_cache_rows']].clone()
    require(hashlib.sha256(hot.numpy().tobytes()).hexdigest() == binding['hot_rows_sha256'], 'RevPR Top-K changed')
    progress(folder, 'preloading_cpu_cache', rows=len(hot))
    loader.cpu_backing_buffer(128, len(hot))
    loader.set_cpu_buffer(hot, len(hot))
    del hot
    # Pin no full feature matrix. CPU feature pages are preloaded by BaM above.
    labels_np = np.load(binding['labels'], mmap_mode='r').reshape(-1)
    labels = torch.from_numpy(np.array(labels_np, dtype=np.int64)).cuda()
    train = np.sort(splits('train'))
    require(hashlib.sha256(train.tobytes()).hexdigest() == binding['train_sorted_sha256'], 'Train split changed')
    roots = (np.random.default_rng(0).permutation(train)[:p['smoke_examples']] if smoke else train)
    features = np.load(binding['features'], mmap_mode='r') if smoke else None
    random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed'])
    torch.cuda.manual_seed_all(p['seed']); dgl.seed(p['seed']); dgl.random.seed(p['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SAGE(128, p['hidden'], p['classes'], p['layers'], p['dropout']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), **p['optimizer'])
    model.train()
    sampler = dgl.dataloading.NeighborSampler(p['fanouts'], replace=False)
    dataloader = GIDS_DGLDataLoader(graph, torch.from_numpy(roots.copy()), sampler,
        p['batch_size'], 128, loader, shuffle=not smoke, drop_last=False,
        num_workers=0, use_alternate_streams=False)
    loader.begin_epoch()
    loader.reset_device_io_stats()
    before = snapshot(loader)
    require(before['gpu']['capacity_pages'] == 8388608, 'Expected 4 GiB of 512-B cache lines')
    sample_start = get_sample_record(); fetch_start = get_fetch_feature_record()
    progress(folder, 'training', smoke=smoke, expected_updates=math.ceil(len(roots)/1024))
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    begin = time.perf_counter_ns()
    model_ns = 0; losses = []; outputs = []; shapes = []; audits = []
    for i, (inp, seeds, blocks, x) in enumerate(dataloader):
        if smoke:
            verify_blocks(blocks, arrays, p['fanouts'])
            reference = np.asarray(features[inp.cpu().numpy()])
            require(np.array_equal(x.cpu().numpy().view('u4'), reference.view('u4')),
                    'Native 512-B feature read differs from original data')
        model_start = time.perf_counter_ns()
        logits = model(blocks, x)
        loss = torch.nn.functional.cross_entropy(logits, labels[seeds])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model_ns += time.perf_counter_ns() - model_start
        losses.append(loss.detach()); outputs.append(seeds.detach())
        shapes.append(dict(input_nodes=len(inp), output_nodes=len(seeds), block_edges=[b.num_edges() for b in blocks]))
        if smoke:
            require(torch.isfinite(logits).all().item(), 'Nonfinite smoke prediction')
            audits.append(dict(batch=i, bit_exact=True, sampled_edges_exact=True, output_nodes=len(seeds)))
        if (i+1) % 100 == 0:
            progress(folder, 'training', smoke=smoke, updates=i+1, elapsed_seconds=(time.perf_counter_ns()-begin)/1e9)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns() - begin) / 1e9
    after = snapshot(loader)
    sample_end = get_sample_record(); fetch_end = get_fetch_feature_record()
    values = torch.stack(losses).cpu().numpy()
    seen = torch.cat(outputs).cpu().numpy()
    require(np.isfinite(values).all(), 'Nonfinite loss')
    finite_parameters = all(torch.isfinite(x).all().item() for x in model.parameters())
    coverage = np.array_equal(np.sort(seen), np.sort(roots))
    feature_access = {k:after['feature'][k]-before['feature'][k] for k in before['feature']}
    gpu = dict(after['gpu'])
    for k in ('requests','hits','inserts','evictions','physical_inserts','partial_fills'):
        if k in gpu:
            gpu[k] -= before['gpu'].get(k, 0)
    gpu['hit_rate'] = gpu['hits']/gpu['requests'] if gpu['requests'] else 0.0
    sample_ns = sample_end[1] - sample_start[1]
    fetch_ns = fetch_end[1] - fetch_start[1]
    timing = dict(e2e_seconds=elapsed, setup_seconds=setup_seconds,
        sample_host_seconds=sample_ns/1e9, fetch_inclusive_host_seconds=fetch_ns/1e9,
        model_host_seconds=model_ns/1e9, native_merged_read_seconds=loader.read_host_ns/1e9,
        window_hint_seconds=loader.pipeline.hint_ns/1e9,
        legacy_overlapping_sum_seconds=(sample_ns+fetch_ns+model_ns)/1e9,
        primary='e2e_seconds', legacy_sum_is_wall_time=False,
        spans_are_host_measurements_not_standalone_cuda_kernel_times=True)
    report = dict(passed=True, smoke=smoke, examples=len(seen), updates=len(shapes),
        tail_size=shapes[-1]['output_nodes'], root_coverage_exact=coverage,
        root_order_sha256=hashlib.sha256(seen.tobytes()).hexdigest(), losses=values.tolist(), shapes=shapes,
        finite_parameters=finite_parameters, pipeline=loader.pipeline.summary(),
        sample_count=sample_end[0]-sample_start[0], fetch_count=fetch_end[0]-fetch_start[0],
        feature_access=feature_access, logical_feature_requests=sum(s['input_nodes'] for s in shapes),
        gpu_cache=gpu, device_io=after['device'], timing=timing,
        smoke_checks=dict(features_bit_exact=bool(smoke and audits), sampled_edges_exact=bool(smoke and audits)),
        audits=audits, protocol=p, candidate_sha256=execution, input_binding_sha256=sha(OUT/'inputs.json'),
        binary_sha256=sha(BINARY), raw_ssd_writes=False, validation=None, test=None,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(folder / 'report.json', report)
    validate_completion(report)
    check_inputs(binding)
    require(verify() == execution, 'Candidate changed during training')
    write(folder / 'accepted.json', dict(passed=True, report_sha256=sha(folder/'report.json'),
        candidate_sha256=execution, binary_sha256=sha(BINARY)))
    progress(folder, 'complete', smoke=smoke, updates=len(shapes), e2e_seconds=elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    # Cooperates with all previous native PA candidates sharing /dev/libnvm0.
    with open('/tmp/digit-pa-sage-libnvm0.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output, args.smoke)


if __name__ == '__main__':
    main()
