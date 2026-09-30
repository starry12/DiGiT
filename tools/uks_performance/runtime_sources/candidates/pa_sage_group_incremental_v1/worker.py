"""A fresh native process, either tail-sensitive smoke or full official epoch."""
import argparse
import fcntl
import math
import random
import resource
from .common import *
from candidates.pa_sage_512b_pair_v4.payload import validate_payload
from candidates.pa_sage_cpu_affinity_v1 import affinity as cpu_affinity
from candidates.pa_sage_512b_pair_v1.runtime_helpers import load_graph, load_bundle, admission, install_cache, verify_digit_blocks


def snapshot(loader):
    return dict(feature=loader.get_feature_access_stats(), gpu=loader.get_gpu_cache_stats(),
                device=loader.get_device_io_stats())


def run(folder, smoke, group_mode, trace=False):
    dense_mode = "graph"
    variant = "overlap"
    arm = 'digit'
    changes = dict(frontier=True,functional=True)
    p = read(HERE / 'protocol.json')
    execution = verify()
    binding = read(OUT / 'inputs.json')
    check_inputs(binding)
    payload_extent=validate_payload(binding,p,arm)
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
    torch.set_num_threads(p['cpu_threads'])
    import BAM_Feature_Store
    from GIDS import GIDS_DGLDataLoader, get_sample_record, get_fetch_feature_record
    from candidates.pa_sage_512b_pair_v4.loader import BoundedGIDS
    from candidates.pa_sage_512b_overlap_v1.loader import OverlapGIDS, batches as scheduled_batches
    LegacyAlignedGIDS = BoundedGIDS if variant=='legacy' else OverlapGIDS
    from candidates.pa_sage_legacy_align_v1.graph import verify_blocks
    from digit.sampler import DiGiTNeighborSampler, DIGIT_STORAGE_ROW
    from models import SAGE
    from ae.pa_sage.common import splits
    require(Path(__import__('sys').modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve() == BINARY,
            'Wrong native binary imported')
    require(getattr(BAM_Feature_Store,'FIFO_WINDOW_512B_API',0)==1,'Wrong FIFO/window backend')
    require(getattr(BAM_Feature_Store,'ASYNC_FEATURE_READ_API',0)==1,'Wrong async backend')
    require(Path(__import__('sys').modules['GIDS.GIDS'].__file__).resolve()==ROOT/'candidates/pa_sage_512b_pair_v3/runtime/GIDS/GIDS.py','Wrong GIDS adapter')
    if changes['frontier']:
        from .sampler import IncrementalGroupSampler
        DiGiTNeighborSampler = IncrementalGroupSampler
    import DiGiTGroupIncrementalCUDA
    require(Path(DiGiTGroupIncrementalCUDA.__file__).resolve()==SAMPLER_BINARY,"Wrong group extension")
    require(DiGiTGroupIncrementalCUDA.INCREMENTAL_GROUP_API==1,"Wrong incremental group API")
    if changes['functional']:
        from candidates.pa_sage_dense_graph_v1.model import DenseGraphSAGE, prepare_model
        SAGE = DenseGraphSAGE
    plan=admission(p,read(Path(binding['bundle'])/'manifest.json'),arm)
    # An extra current/prefetched batch exists outside the original WB queue.
    # Include block/feature/activation headroom, not just one feature matrix.
    plan['components_bytes']['overlap_additional_batch_allowance']=512*2**20
    plan['required_bytes']+=math.ceil(512*2**20*1.15)
    plan['components_bytes']['dense_graph_pool_allowance']=256*2**20
    plan['required_bytes']+=math.ceil(256*2**20*1.15)
    free,total=torch.cuda.mem_get_info()
    require(free>=plan['required_bytes'], 'GPU memory admission failed: free=%d required=%d' % (free,plan['required_bytes']))
    folder.mkdir(parents=True, exist_ok=False)
    write(folder/'admission.json',dict(plan,free_bytes=free,total_bytes=total,passed=True))
    setup_start = time.perf_counter()
    progress(folder, 'loading_directed_graph', smoke=smoke)
    graph, arrays = load_graph(binding, p['nodes'], p['edges'])
    bundle=load_bundle(binding) if arm=='digit' else None
    sampler=(DiGiTNeighborSampler(p['fanouts'],bundle,cuda_mode='required',
                metadata_mode=p['digit_metadata_mode'],random_seed=p['seed'],group_mode=group_mode) if bundle else
                dgl.dataloading.NeighborSampler(p['fanouts'],replace=False))
    if bundle:
        progress(folder,'uploading_compact_sampler_metadata')
        sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    progress(folder, 'initializing_512b_cache')
    payload=binding['digit_payload'] if bundle else binding['ssd_payload']
    storage_rows=bundle.manifest['feature']['num_storage_rows'] if bundle else p['nodes']
    require(storage_rows==payload_extent['runtime_rows'],'Runtime extent differs from validated payload')
    loader = LegacyAlignedGIDS(page_size=512, off=payload['offset'], cache_dim=128,
        num_ele=storage_rows*128, num_ssd=1, ssd_list=[0], cache_size=p['cache_mib'], ctrl_idx=0,
        window_buffer=True, wb_size=2, accumulator_flag=True, feature_index_mode=p['feature_index_modes'][arm],
        gpu_cache_policy=p['gpu_policies'][arm], cpu_feature_path='mapped', mixed_io=False, device_io_stats=True)
    loader.set_required_storage_access(**p['accumulator_parameters'])
    require(math.isclose(loader.required_accesses,p['expected_required_accesses']),'Accumulator threshold changed')
    progress(folder,'preloading_cpu_cache',arm=arm,rows=p['cpu_rows'])
    cpu_cache,cache_refs=install_cache(loader,arm,binding,bundle,p)
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
    initial=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        initial.update(name.encode());initial.update(tensor.detach().cpu().numpy().tobytes())
    initial_model_sha256=initial.hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), **p['optimizer'])
    model.train()
    progress(folder, 'preparing_last_dense', dense_mode=dense_mode)
    preparation=prepare_model(model,optimizer,dense_mode,batch_size=p['batch_size'])
    require(preparation['retained_allocation_bytes']<=256*2**20,'Dense pool exceeded admission allowance')
    write(folder/'dense_preparation.json',preparation)
    dataloader = GIDS_DGLDataLoader(graph, torch.from_numpy(roots.copy()), sampler,
        p['batch_size'], 128, loader, shuffle=not smoke, drop_last=False,
        num_workers=0, use_alternate_streams=False)
    loader.begin_epoch()
    loader.reset_device_io_stats()
    before = snapshot(loader)
    require(before['gpu']['capacity_pages'] == 8388608, 'Expected 4 GiB of 512-B cache lines')
    sample_start = get_sample_record(); fetch_start = get_fetch_feature_record()
    affinity_record = cpu_affinity.apply('one_core', p['bound_logical_cpu'])
    conversion_ns = 0; model_cpu_ns = 0
    forward_ns = 0; backward_ns = 0; optimizer_ns = 0; loss_ns = 0
    observed_block_dtypes = set()
    progress(folder, 'training', smoke=smoke, arm=arm, affinity='one_core', expected_updates=math.ceil(len(roots)/1024))
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    epoch_cpu_start = time.process_time_ns()
    begin = time.perf_counter_ns()
    model_ns = 0; losses = []; outputs = []; shapes = []; audits = []
    profiler = None
    if trace:
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2,warmup=1,active=16,repeat=1),
            on_trace_ready=lambda p:p.export_chrome_trace(str(folder/'cuda_trace.json')),
            record_shapes=False,profile_memory=False,with_stack=False)
        profiler.__enter__()
    try:
        for i, (inp, seeds, blocks, x) in enumerate(scheduled_batches(dataloader,loader,variant)):
            if smoke:
                (verify_digit_blocks if bundle else verify_blocks)(blocks, arrays, p['fanouts'])
                if bundle:
                    rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
                    require(np.all((rows>=0)&(rows<storage_rows)),'Storage row out of range')
                    require(np.array_equal(bundle.arrays['storage_to_node'][rows],inp.cpu().numpy()),'Storage inverse differs')
                expected_features = np.asarray(features[inp.cpu().numpy()])
                require(np.array_equal(x.cpu().numpy().view('u4'), expected_features.view('u4')),
                        'Native 512-B feature read differs from original data')
            conversion_start = time.perf_counter_ns()
            blocks = [block.int().to(x.device) for block in blocks]
            observed_block_dtypes.update(str(block.idtype) for block in blocks)
            conversion_ns += time.perf_counter_ns() - conversion_start
            cpu_start = time.process_time_ns()
            model_start = time.perf_counter_ns()
            logits = model(blocks, x)
            part = time.perf_counter_ns(); forward_ns += part - model_start
            loss = torch.nn.functional.cross_entropy(logits, labels[seeds])
            optimizer.zero_grad(set_to_none=True)
            end = time.perf_counter_ns(); loss_ns += end - part
            loss.backward()
            part = time.perf_counter_ns(); backward_ns += part - end
            optimizer.step()
            end = time.perf_counter_ns(); optimizer_ns += end - part
            model_ns += end - model_start
            model_cpu_ns += time.process_time_ns() - cpu_start
            losses.append(loss.detach()); outputs.append(seeds.detach())
            shapes.append(dict(input_nodes=len(inp), output_nodes=len(seeds), block_edges=[b.num_edges() for b in blocks]))
            if smoke:
                require(torch.isfinite(logits).all().item(), 'Nonfinite smoke prediction')
                audits.append(dict(batch=i, bit_exact=True, sampled_edges_exact=True, output_nodes=len(seeds)))
            if profiler is not None:profiler.step()
            if (i+1) % 100 == 0:
                progress(folder, 'training', smoke=smoke, updates=i+1, elapsed_seconds=(time.perf_counter_ns()-begin)/1e9)
    except BaseException as error:
        write(folder/'runtime_failure.json',dict(passed=False,error=str(error),completed_updates=len(shapes),
            pipeline=loader.pipeline.summary() if loader.pipeline else None,last_read=loader.last_read,
            torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved()))
        progress(folder,'failed',arm=arm,smoke=smoke,updates=len(shapes),error=str(error))
        cleanup_errors=[]
        if variant!='legacy':
            try:loader.drain()
            except BaseException as cleanup:cleanup_errors.append(str(cleanup))
        if profiler is not None:
            try:profiler.__exit__(type(error),error,error.__traceback__)
            except BaseException as cleanup:cleanup_errors.append(str(cleanup))
        if cleanup_errors:write(folder/'cleanup_failure.json',dict(original_error=str(error),cleanup_errors=cleanup_errors))
        raise
    if profiler is not None:profiler.__exit__(None,None,None)
    if variant!='legacy':loader.drain()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns() - begin) / 1e9
    epoch_cpu_ns = time.process_time_ns() - epoch_cpu_start
    affinity_record['at_training_end'] = cpu_affinity.thread_masks()
    require(observed_block_dtypes == {'torch.int32'}, 'Model blocks were not int32')
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
        ssd_active_seconds=after['device']['active_ns']/1e9,
        block_conversion_host_seconds=conversion_ns/1e9,
        model_process_cpu_seconds=model_cpu_ns/1e9, epoch_process_cpu_seconds=epoch_cpu_ns/1e9,
        forward_host_seconds=forward_ns/1e9, loss_zero_grad_host_seconds=loss_ns/1e9,
        backward_host_seconds=backward_ns/1e9, optimizer_host_seconds=optimizer_ns/1e9,
        cpu_observation_host_seconds=0.0,
        sample_host_seconds=sample_ns/1e9, fetch_inclusive_host_seconds=(fetch_ns/1e9 if variant=='legacy' else loader.fetch_host_ns/1e9),
        model_host_seconds=model_ns/1e9, native_merged_read_seconds=loader.read_host_ns/1e9,
        window_hint_seconds=loader.pipeline.hint_ns/1e9,
        legacy_overlapping_sum_seconds=(sample_ns+fetch_ns+model_ns)/1e9,
        primary='e2e_seconds', legacy_sum_is_wall_time=False,
        spans_are_host_measurements_not_standalone_cuda_kernel_times=True)
    final_state=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        final_state.update(name.encode());final_state.update(tensor.detach().cpu().numpy().tobytes())
    report = dict(variant=group_mode,group_mode=group_mode,incremental_group_api=1,
        sampler_binary_sha256=sha(SAMPLER_BINARY),dense_mode=dense_mode,io_mode=variant,eid_mode="original",
        dense_graph=model.dense_summary(),diagnostic_trace=trace,
        async_backend=(dict(outstanding=0) if variant=='legacy' else loader.async_summary()),changes=changes,additional_sampling_profile=False,
        model_implementation='last_dense_cuda_graph' if dense_mode=='graph' else 'functional_mean_sage',
        frontier_implementation='known_extent_coo' if changes['frontier'] else 'dgl_graph',
        final_model_sha256=final_state.hexdigest(),passed=True, smoke=smoke, examples=len(seen), updates=len(shapes),
        tail_size=shapes[-1]['output_nodes'], root_coverage_exact=coverage,
        root_order_sha256=hashlib.sha256(seen.tobytes()).hexdigest(), losses=values.tolist(), shapes=shapes,
        finite_parameters=finite_parameters, pipeline=loader.pipeline.summary(),
        sample_count=sample_end[0]-sample_start[0], fetch_count=fetch_end[0]-fetch_start[0],
        feature_access=feature_access, logical_feature_requests=sum(s['input_nodes'] for s in shapes),
        gpu_cache=gpu, device_io=after['device'], timing=timing,
        fifo_window_512b_api=1,
        fifo_window_policy='ring ticket with protected/busy victim skipping at 512 B',
        payload_extent=payload_extent, arm=arm, affinity=affinity_record, block_index_bits=32, continuous_cpu_frequency_observation=False,
        initial_model_sha256=initial_model_sha256,
        page_bytes=512, window_buffer=True, accumulator=True, mixed_io=False,
        cpu_cache_rows=p['cpu_rows'],cpu_cache=cpu_cache,
        native_grouped_batches=sampler._cuda_call_counter if bundle else 0,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        torch_version=torch.__version__, dgl_version=dgl.__version__,
        smoke_checks=dict(features_bit_exact=bool(smoke and audits), sampled_edges_exact=bool(smoke and audits),
                          storage_inverse_exact=bool(smoke and bundle and audits)),
        audits=audits, protocol=p, candidate_sha256=execution, input_binding_sha256=sha(OUT/'inputs.json'),
        binary_sha256=sha(BINARY), raw_ssd_writes=False, validation=None, test=None,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(folder / 'report.json', report)
    validate_completion(report)
    check_inputs(binding)
    require(verify() == execution, 'Candidate changed during training')
    write(folder / 'accepted.json', dict(passed=True, report_sha256=sha(folder/'report.json'),
        candidate_sha256=execution, binary_sha256=sha(BINARY),sampler_binary_sha256=sha(SAMPLER_BINARY)))
    progress(folder, 'complete', smoke=smoke, updates=len(shapes), e2e_seconds=elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--trace',action='store_true')
    args = parser.parse_args()
    # Cooperates with all previous native PA candidates sharing /dev/libnvm0.
    with open('/tmp/digit-pa-sage-libnvm0.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output, args.smoke, args.variant,args.trace)


if __name__ == '__main__':
    main()
