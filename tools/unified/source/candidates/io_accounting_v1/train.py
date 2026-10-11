"""Readiness checks and paired no-BFS PA/SAGE training."""
import math
from pathlib import Path
import time
from ae.pa_sage.common import *
setup_imports()


def readiness(data, live=False):
    from ae.common import check_payload
    from digit.gpu_admission import check_live
    p = cfg()
    if not (Path(data)/'prepared.json').exists():
        stages=['sources.json','bootstrap/validation.json','hot_ready.json','final/validation.json',
                'traces_ready.json','final/payload_ready.json','admission.json','prepared.json']
        return dict(passed=True,protocol_sha256=sha(PROTOCOL),bfs_enabled=False,
                    filesystem_inputs_ready=False,smoke_ready=False,full_training_ready=False,
                    admission_queried_live=False,fresh_raw_ssd_read_performed=False,
                    missing=[name for name in stages if not (Path(data)/name).exists()])
    ready = check_prepared(data)
    m = read(Path(data) / 'final/bundle/manifest.json')
    require(m['grouping']['group_size'] == p['paper_specified']['group_size'], 'Wrong group size')
    require(m['metadata']['pa_sage_random_v2']['protocol_sha256'] == sha(PROTOCOL), 'Bundle protocol mismatch')
    require(read(Path(data) / 'orders.json')['bfs_enabled'] is False, 'BFS order forbidden')
    for split in ('valid', 'test'):
        trace = read(Path(data) / (split + '_trace/manifest.json'))
        require(trace['sampling']['fanouts'] == p['paper_specified']['fanouts'], 'Wrong trace fanout')
        require(trace['source']['protocol_sha256'] == sha(PROTOCOL), 'Wrong trace protocol')
    from ae.pa_sage.admission import audited_plan
    from ae.pa_sage.native_gate import check_native
    plan = audited_plan(Path(data))
    # The frozen 32-byte logical page ABI is unchanged. Charge one 8-byte
    # bitmap per physical cache slot plus six uint64 counters explicitly.
    extra=p['reconstruction_choices']['gpu_cache_bytes']//p['reconstruction_choices']['io_page_bytes']*8+48
    plan['base_schema']=plan['schema']
    plan['schema']='digit-useful-io-admission-v1'
    plan['base_required_bytes']=plan['required_bytes']
    plan['components_bytes']['useful_io_bitmap_and_counters']=extra
    plan['required_bytes']+=math.ceil(extra*plan.get('safety_multiplier',1.2))
    plan['useful_io_metadata_bytes']=extra
    if live:
        plan = check_live(plan)
    baseline = check_payload('papers_gids')
    receipt_path = Path(data) / 'ssd_ready.json'
    full = None
    if receipt_path.exists():
        from digit.ssd_payload import validate_bundle_verify_receipt
        from digit.artifacts import load_artifact_bundle
        full = read(receipt_path)
        require(full['protocol_sha256'] == sha(PROTOCOL), 'SSD receipt protocol mismatch')
        require(full['manifest_sha256'] == sha(Path(data) / 'final/bundle/manifest.json'), 'SSD manifest mismatch')
        require(full['full_readback_passed'], 'SSD full readback missing')
        state = read(full['state'])
        require(sha(full['state']) == full['state_sha256'] and state['status'] == 'verified', 'SSD state changed')
        require(state['feature_file_sha256'] == m['files']['reordered_features']['sha256'], 'SSD payload file changed')
        require(state['device_offset_bytes'] == full['offset'] and state['payload_bytes'] == m['feature']['num_storage_rows']*512, 'SSD range mismatch')
        bundle = load_artifact_bundle(Path(data) / 'final/bundle', validation_mode='fast')
        require(sha(full['verify_receipt']) == full['verify_receipt_sha256'], 'SSD readback receipt changed')
        validate_bundle_verify_receipt(full['verify_receipt'], bundle, device_offset_bytes=full['offset'])
    missing = []
    if not live:
        missing.append('Live GPU and host admission has not been queried by this check.')
    elif not plan['passed']:
        missing.append('The current GPU or host memory budget does not pass training admission.')
    if full is None:
        missing.append('New g2 SSD payload has not been written and fully read back.')
    native = check_native(data) if full is not None else dict(passed=False, missing=['SSD prerequisite missing'])
    smoke_ready = bool(live and plan['passed'] and full is not None)
    missing.extend(native.get('missing', []))
    return dict(passed=True, protocol_sha256=sha(PROTOCOL), bfs_enabled=False,
                paired_random_order=True, filesystem_inputs_ready=True,
                admission=plan, admission_queried_live=live,
                baseline_historical_receipt_valid=True, baseline=baseline, full=full,
                fresh_raw_ssd_read_performed=False,
                smoke_ready=smoke_ready, native_validation=native,
                full_training_ready=bool(smoke_ready and native['passed']),
                missing=missing)


def run(args):
    import fcntl
    require(__debug__, 'Python optimization would disable correctness assertions')
    with open('/tmp/digit-pa-sage-libnvm0.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(args)


def _run(args):
    from candidates.io_accounting_v1.common import verify_release
    from ae.common import check_device
    require(args.smoke, "This candidate entry is a short I/O gate, not the selected bidirectional experiment")
    data = Path(args.data).resolve()
    p = cfg(); choices = p['reconstruction_choices']
    require(args.seed in choices['seeds'], 'Unsupported seed')
    execution = verify_release()
    state = readiness(data, live=True)
    gate = 'smoke_ready' if args.smoke else 'full_training_ready'
    require(state[gate], 'PA training blocked: '+str(state['missing']))
    check_device()
    ssd_binding = dict(ready_sha256=sha(data/'ssd_ready.json'), baseline_state_sha256=sha(state['baseline']['state']),
                       full_state_sha256=sha(state['full']['state']), full_offset=state['full']['offset'])
    output = Path(args.output).resolve()
    require(not output.parent.exists(), 'Use a fresh output directory')
    output.parent.mkdir(parents=True)
    from contextlib import nullcontext
    from ae.pa_sage.resources import Observations
    observer = Observations(output.parent/'resources.json')
    with observer if observer is not None else nullcontext():
        return _execute(args, data, p, execution, state, ssd_binding, output, observer)


def _execute(args, data, p, execution, state, ssd_binding, output, observer):
    import numpy as np
    import torch
    import runner as r
    from candidates.io_accounting_v1.common import verify_release
    from digit.artifacts import load_artifact_bundle
    from GIDS import GIDS
    from candidates.io_accounting_v1 import training_loop
    from ae.pa_sage.admission import PROFILE
    choices = p['reconstruction_choices']
    def update(**kw):
        write(output.parent/'progress.json',dict(updated_unix=time.time(), **kw))
        print(__import__('json').dumps(kw), flush=True)
    def mark(stage):
        if observer is not None:
            observer.mark(stage)
        update(stage=stage, arm=args.arm, seed=args.seed, smoke=args.smoke)
    start = time.perf_counter(); r.startup()
    bundle = load_artifact_bundle(data/'final/bundle', validation_mode='fast') if args.arm == 'digit_full' else None
    cpu_rows = array(data/('full_cpu_rows.npy' if bundle else 'gids_cpu_rows.npy'))
    payload = state['full'] if bundle else state['baseline']
    offset = payload['offset']
    cache_mib = choices['gpu_cache_bytes']//2**20
    rows = bundle.manifest['feature']['num_storage_rows'] if bundle else payload['bytes']//512
    if not bundle:
        require(0 <= rows-source_config()['num_nodes'] < choices['io_page_bytes']//512, 'Unexpected baseline padding')
    from digit.gpu_admission import check_live
    state['admission'] = check_live(state['admission'])
    require(state['admission']['passed'], 'Memory availability changed before native allocation')
    mark('initializing_native_cache')
    loader = GIDS(page_size=choices['io_page_bytes'], off=offset, cache_dim=128,
            num_ele=rows*128, num_ssd=1, ssd_list=[0], cache_size=cache_mib,
            ctrl_idx=0, window_buffer=False, accumulator_flag=False,
            feature_index_mode='explicit' if bundle else 'logical',
            gpu_cache_policy='fifo' if bundle else 'legacy', cpu_feature_path='mapped',
            mixed_io=bool(bundle), device_io_stats=True,
            mixed_io_geometry=bundle.io_geometry.with_payload_offset(offset).to_native_mapping() if bundle else None)
    mark('native_cache_allocated')
    require(len(cpu_rows) == choices['cpu_cache_rows'], 'CPU cache budget mismatch')
    loader.cpu_backing_buffer(128, len(cpu_rows))
    loader.set_cpu_buffer(torch.from_numpy(np.asarray(cpu_rows).copy()), len(cpu_rows))
    torch.cuda.synchronize(); loader.reset_device_io_stats()
    require(loader.get_gpu_cache_stats()['capacity_pages'] == choices['gpu_cache_bytes']//choices['io_page_bytes'], 'GPU cache budget mismatch')
    mark('cpu_preloaded')
    g = graph()
    mark('graph_loaded')
    labels = array(source_config()['label_identity']['path']).reshape(-1)
    ptr = array(csc_paths()[0]); degrees_np = np.diff(ptr)
    require(degrees_np.max() <= 2**31-1, 'Degree counter overflow')
    degrees = torch.from_numpy(degrees_np.astype(np.int32)).cuda(); del degrees_np
    mark('degrees_loaded')
    def progress(**kw):
        if observer is not None:
            observer.mark(kw['stage']+'_epoch_'+str(kw.get('epoch', 0)))
        update(**kw)
    training_loop.progress = progress
    setup_seconds = time.perf_counter()-start
    features = array(source_config()['source_features']['path']) if args.smoke else None
    result = training_loop.run_training(args.arm,args.seed,loader,bundle,g,labels,degrees,
                                       output,data,p,smoke=args.smoke,features=features)
    mark('training_complete')
    expected = 4 if args.smoke else p['paper_specified']['epochs']*math.ceil(len(splits('train'))/p['paper_specified']['batch_size'])
    require(result['updates'] == expected, 'Unexpected training length')
    require(verify_release() == execution, 'Source release changed during run')
    require(sha(data/'ssd_ready.json') == ssd_binding['ready_sha256'], 'SSD receipt changed during run')
    require(sha(state['baseline']['state']) == ssd_binding['baseline_state_sha256'] and
            sha(state['full']['state']) == ssd_binding['full_state_sha256'], 'SSD state changed during run')
    trace_seconds=result['evaluation_lifecycle']['trace_preparation_seconds']
    online=result['training_seconds']+result['validation_seconds']+(result['test']['seconds'] if result['test'] else 0.)
    result['timing_totals']=dict(trace_preparation_seconds=trace_seconds,training_validation_test_seconds=online,with_trace_preparation_seconds=online+trace_seconds,diagnostic_control_seconds=0.)
    result.update(execution_profile_sha256=sha(ROOT/'configs/paper/pa_sage_execution_v4.json'),repeat=getattr(args,'repeat',0),admission_profile_sha256=sha(PROFILE), cpu_cache_rows=len(cpu_rows),
                  protocol_id=p['protocol_id'], protocol_sha256=sha(PROTOCOL), execution_sha256=execution,
                  prepared_sha256=sha(data/'prepared.json'), bfs_enabled=False, paired_random_order=True,
                  cache_bytes=choices['gpu_cache_bytes'], setup_seconds=setup_seconds,
                  worker_seconds=time.perf_counter()-start,
                  mean_training_epoch_seconds=result['training_seconds']/len(result['epochs']),
                  admission=state['admission'], ssd_binding=ssd_binding, paper_original_version=False, raw_writes=False)
    if observer is not None:
        result['resource_observations'] = observer.result()
    result['native_validation_at_start'] = state['native_validation']
    result['native_validation_at_start_is_frozen_base_only']=True
    from candidates.io_accounting_v1.accounting import summarize
    result['io_accounting_training']=summarize([(e['training'],e['train_seconds']) for e in result['epochs']])
    result['io_accounting_scope']='short directed PA gate, not selected bidirectional accuracy/performance'
    result['io_accounting_metadata_bytes']=choices['gpu_cache_bytes']//choices['io_page_bytes']*8+48
    result['execution_profile_is_frozen_base_only']=True
    write(output,result)
    write(output.parent/'worker_ready.json',dict(passed=True,report_sha256=sha(output)))
    deadline=time.monotonic()+120
    while not (output.parent/'release_worker.json').exists():
        require(time.monotonic()<deadline,'Controller monitor shutdown handshake timed out')
        time.sleep(.25)

