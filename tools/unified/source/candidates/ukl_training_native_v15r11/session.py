"""Production lifecycle callable by the subsequent bounded smoke controller.

This module deliberately supplies no unguarded full-graph execute CLI. The
factory verifies effective cgroup limits and accepted SSD before GPU use.
"""
import gc,os,time
from pathlib import Path
import numpy as np
from candidates.ukl_runtime_prepare_v11r1 import protocol as U
from candidates.ukl_training_adapter_v12.inputs import accepted_inputs,arm_inputs,load_window
from candidates.ukl_native_sampling_v10r4 import sampling as S,fork_guard as H
from candidates.ukl_real_multi_v9.memory import Arena,Registration
from candidates.ukl_native_sampling_v10r4.worker import progress_callback
from candidates.ukl_training_native_v15r11.storage import accepted_storage
from .protocol import verify_manifest,require_admitted_worker
from .backend import prepare_dependencies,Provider
from .training import prepare_gpu_model,Engine


def run_arm(arm,mode,*,check,event=lambda *a,**kw:None):
    if mode not in ('smoke','performance'):raise ValueError('Unknown training mode')
    b=require_admitted_worker(arm);digest=verify_manifest();storage=accepted_storage()
    # Controller holds shared AE/author locks; identity check happens before
    # ownership because the legacy identify helper spawns a short process.
    from ae.common import check_device
    check_device();inputs=arm_inputs(accepted_inputs(),arm);roots,labels=load_window(inputs)
    module=prepare_dependencies();model,opt=prepare_gpu_model()
    import torch
    if torch.cuda.mem_get_info()[0]<b['gpu_free_min']:raise RuntimeError('GPU free-memory admission below 24 GiB')
    def admit(remaining=0):
        check();available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        if available<remaining+b['host_reserve']:raise RuntimeError('Remaining host allocation admission')
    admit(b['host_min']-b['host_reserve'])
    specs=U.BASE.binding();extra=b['host_cgroup_max']-b['host_components']['graph_arena']
    guard=H.OwnershipGuard().enter();arena=aux=graph=registration=native=provider=engine=None;hot=inverse=None
    try:
        arena=Arena(specs,admission_check=lambda rem:admit(rem+extra),max_bytes=b['host_components']['graph_arena'])
        guard.track(arena);H.dontfork_arena(arena)
        arena.load(progress_callback(event,lambda:admit(arena.remaining_bytes+extra),512*2**20))
        H.assert_dontfork(arena);graph=S.Graph(arena,U.METADATA)
        sources={'hot':inputs['hot']}
        if arm=='digit':sources['inverse']=accepted_inputs()['stages']['graph']['files']['storage_to_node.i32']
        auxspec={k:dict(path=v['path'],offset=0,length=v['bytes'],identity=v['identity'],sha256=v['sha256']) for k,v in sources.items()}
        aux=Arena(auxspec,admission_check=lambda rem:admit(rem+extra),max_bytes=8*2**30)
        guard.track(aux);H.dontfork_arena(aux);aux.load(progress_callback(event,lambda:admit(extra),512*2**20));H.assert_dontfork(aux)
        hot=np.frombuffer(aux.mm,np.int64,count=inputs['cpu_cache_rows'],offset=aux.offsets['hot']);hot.flags.writeable=False
        if arm=='digit':inverse=np.frombuffer(aux.mm,np.int32,count=graph.storage_rows,offset=aux.offsets['inverse']);inverse.flags.writeable=False
        registration=Registration(arena);H.assert_dontfork(arena);native=S.Native(graph,registration)
        admit(b['host_components']['cpu_features']+b['host_components']['backend_page_initialization']+b['host_components']['logical_slot_lookup']+16*2**30)
        provider=Provider(module,storage,arm,graph,hot,inverse,lambda:admit(16*2**30))
        engine=Engine(native,arm,model,opt,provider,lambda:admit(8*2**30))
        result=engine.window(roots[:4],labels[:4],0,4,event=lambda **kw:event(**kw)) if mode=='smoke' else engine.window(roots,labels,event=lambda **kw:event(**kw))
        result.update(cache=provider.cache,arm=arm,mode=mode,manifest_sha256=digest,storage_acceptance_sha256=storage['acceptance_sha256'])
    finally:
        if provider is not None:provider.close()
        engine=provider=None;hot=inverse=None
        if native is not None:native.close()
        if registration is not None:registration.close()
        if graph is not None:graph.close()
        native=registration=graph=None;gc.collect()
        for a in (aux,arena):
            if a is not None:a.close();guard.confirm_released(a)
        guard.release()
    if verify_manifest()!=digest:raise RuntimeError('Training source changed')
    result.update(graph_and_aux_released=True,ownership=guard.receipt(),process_exit_acceptance_required=True)
    return result
