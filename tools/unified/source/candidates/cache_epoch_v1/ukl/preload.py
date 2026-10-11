"""Real SSD prefix preload with full value checks, no graph and no training."""
import hashlib,os,time
from pathlib import Path
import numpy as np
from . import protocol as P
from .phase import create_probe,await_phase,verify_charge_handoff
from candidates.ukl_training_native_v15r11.backend import prepare_dependencies,RETAINED
from candidates.ukl_training_native_v15r11.external import ExternalCache
from .storage import accepted_storage
from candidates.ukl_training_native_v15r11.gpu_identity import verify_cuda_uuid
from candidates.ukl_real_multi_v9.memory import Arena,CudaAPI
from candidates.ukl_native_sampling_v10r4 import fork_guard as H
from .inputs import accepted_inputs
from candidates.ukl_runtime_prepare_v11r1.data import features
from candidates.pa_sage_cache_policy_v6.counters import snapshot
from .lifecycle import record_failure,cleanup_failed,raise_failures

ROWS=65536;EXTENT=131072;GPU_BYTES=16*2**20

def run(arm,check,event):
    P.require_admitted_worker(arm);digest=P.verify_manifest();storage=accepted_storage()
    from ae.common import check_device
    check_device();inputs=accepted_inputs();module=prepare_dependencies()
    import torch
    from .cuda_context import initialize
    context=initialize(P.SELECTED[1])
    event('cuda_context_ready',identity=context)
    if torch.cuda.mem_get_info()[0]<24*2**30:raise RuntimeError('GPU free memory admission')
    out=Path(os.environ['UKL_V27_OUTPUT']);probe=create_probe(out)
    event('initialization_ready');ack=await_phase(out,'data',check)
    phase=verify_charge_handoff(out,ack,probe);P.require_admitted_worker(arm)
    region=storage['arms'][arm]['region']
    if region['row_bytes']!=512 or region['payload_bytes']<EXTENT*512 or region['device_offset_bytes']%4096:raise RuntimeError('Accepted SSD prefix extent')
    ctrl=module.GIDS_Controllers();fs=module.BAM_Feature_Store_float();RETAINED.append((fs,ctrl))
    guard=H.OwnershipGuard().enter();aux=cache=None;logical=None
    result=None;failure=cleanup_failure=None
    try:
        if arm=='digit':
            source=inputs['stages']['graph']['files']['storage_to_node.i32']
            spec=dict(path=source['path'],offset=0,length=ROWS*4,identity=source['identity'])
            aux=Arena({'inverse_prefix':spec},admission_check=lambda rem:check(),max_bytes=ROWS*4)
            guard.track(aux);H.dontfork_arena(aux);aux.load(lambda *args:check());H.assert_dontfork(aux)
            logical=np.frombuffer(aux.mm,np.int32,count=ROWS).astype(np.int64)
        else:logical=np.arange(ROWS,dtype=np.int64)
        if np.any(logical<0) or np.any(logical>=inputs['nodes']):raise RuntimeError('Inverse logical bounds')
        check();ctrl.init_GIDS_controllers(1,1024,128,[0])
        fs.init_controllers(ctrl,4096,region['device_offset_bytes']//4096,GPU_BYTES//2**20,EXTENT*128,1,1,0)
        fs.set_mixed_io_geometry(True,512,4096,2,1,1,hashlib.sha256(b'UKL-v15-512B-row-4096B-page-g2-FIFO').hexdigest())
        fs.set_device_io_stats(True);fs.set_cpu_feature_path(0,131072,1)
        event('ssd_preload_begin')
        cache=ExternalCache(fs,np.arange(ROWS,dtype=np.int64),EXTENT,CudaAPI(),check,lambda **k:event(**k))
        # Full 32 MiB reference comparison in <= 512 KiB slices.
        h=hashlib.sha256()
        for lo in range(0,ROWS,1024):
            check();actual=np.frombuffer(cache.pool.mm,np.float32,count=1024*128,offset=lo*512).reshape(1024,128)
            try:
                if not np.array_equal(actual,features(logical[lo:lo+1024])):raise RuntimeError('Real SSD preload feature mismatch')
                h.update(actual.tobytes())
            finally:
                del actual
        slots=np.zeros(EXTENT,np.uint32);slots[:ROWS]=np.arange(1,ROWS+1,dtype=np.uint32)
        fs.policy_write_cpu_map(0,slots);fs.policy_finish_cpu_cache();fs.policy_configure(2,GPU_BYTES)
        counters=snapshot(fs)
        if counters['policy']['preload_rows']!=ROWS or counters['gpu']['resident_pages']!=0 or sum(counters['feature'].values())!=0:raise RuntimeError('Preload state/counter mismatch')
        d=counters['device']
        if d['outstanding'] or d['submitted_commands']<=0 or d['submitted_commands']!=d['completed_commands'] or d['completed_bytes']<=0:raise RuntimeError('Real SSD completion accounting')
        result=dict(passed=True,stage=P.STAGE,arm=arm,mode='ssd_preload',rows=ROWS,values_checked=ROWS*128,
                    feature_sha256=h.hexdigest(),cuda_context=context,inverse_prefix_sha256=aux.digests if aux else None,
                    io=d,updates=0,graph_loaded=False,phase_accounting=phase,
                    manifest_sha256=digest,storage_acceptance_sha256=storage['acceptance_sha256'])
    except BaseException as error:
        failure=record_failure(error)
    finally:
        try:
            if cache is not None:cache.close()
            logical=None
            if aux is not None:aux.close();guard.confirm_released(aux)
            guard.release()
        except BaseException as error:
            cleanup_failure=record_failure(error)
            cleanup_failed(event,failure,cleanup_failure)
    raise_failures(failure,cleanup_failure)
    result.update(cache_released=cache.closed,ownership=guard.receipt());return result
