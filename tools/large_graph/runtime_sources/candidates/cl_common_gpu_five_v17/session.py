"""Production lifecycle callable by the subsequent bounded smoke controller.

This module deliberately supplies no unguarded full-graph execute CLI. The
factory verifies effective cgroup limits and accepted SSD before GPU use.
"""
import gc,os,time
from pathlib import Path
import numpy as np
from . import dataset as U
from .inputs import accepted_inputs,arm_inputs,load_window
from . import sampling as S
from candidates.ukl_native_sampling_v10r4 import fork_guard as H
from candidates.ukl_real_multi_v9.memory import Arena,Registration
from candidates.ukl_native_sampling_v10r4.worker import progress_callback
from .storage import accepted_storage
from .protocol import verify_manifest,require_admitted_worker
from .backend import prepare_dependencies,Provider
from .training import prepare_gpu_model
from .performance_engine import Engine
from . import protocol as P
from .gpu import verify_cuda_uuid
from .lifecycle import record_failure,cleanup_failed,raise_failures
from .numa import bind_fresh,placement


def run_arm(arm,mode,*,check,event=lambda *a,**kw:None):
    if mode != 'performance':raise ValueError('Unknown training mode')
    b=require_admitted_worker(arm);digest=verify_manifest();storage=accepted_storage()
    # Controller holds shared AE/author locks; identity check happens before
    # ownership because the legacy identify helper spawns a short process.
    from ae.common import check_device
    check_device();accepted=accepted_inputs();inputs=arm_inputs(accepted,arm);roots,labels=load_window(inputs)
    module=prepare_dependencies();model,opt=prepare_gpu_model()
    verify_cuda_uuid(P.SELECTED[1])
    import torch
    from .gpu_budget import admit_initialized
    gpu_budget = admit_initialized(b, *torch.cuda.mem_get_info())
    event('gpu_memory_admitted', budget=gpu_budget)
    from .io_probe import run as io_probe
    io_check=io_probe(module,storage,arm,accepted,check,event)
    def admit(remaining=0):
        check();available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        from .host_budget import require_available
        require_available(available,remaining,b['host_reserve'])
    admit(b['host_min']-b['host_reserve'])
    # Move once after initialization; kernel memory charges remain in init.
    from .phase import create_probe,await_phase,verify_charge_handoff
    out_dir=Path(os.environ['CL_V17_OUTPUT'])
    init_probe=create_probe(out_dir)
    event('initialization_ready')
    phase_ack=await_phase(out_dir,'data',check)
    require_admitted_worker(arm)
    phase_evidence=verify_charge_handoff(out_dir,phase_ack,init_probe)
    event('data_accounting_ready',phase=phase_evidence)
    admit(b['host_min']-b['host_reserve'])
    specs=U.BASE.binding();extra=b['host_cgroup_max']-b['host_components']['graph_arena']
    guard=H.OwnershipGuard().enter();arena=aux=graph=registration=native=provider=engine=None;hot=inverse=None
    result=None;failure=cleanup_failure=None
    try:
        arena=Arena(specs,admission_check=lambda rem:admit(rem+extra),max_bytes=b['host_components']['graph_arena'])
        guard.track(arena);H.dontfork_arena(arena)
        if arm=='digit':event('numa_policy_installed',**bind_fresh(arena,'graph'))
        arena.load(progress_callback(event,lambda:admit(arena.remaining_bytes+extra),512*2**20))
        H.assert_dontfork(arena);graph=S.Graph(arena,U.METADATA)
        sources={'hot':inputs['hot']}
        if arm=='digit':sources['inverse']=accepted_inputs()['stages']['graph']['files']['storage_to_node.i32']
        auxspec={k:dict(path=v['path'],offset=0,length=v['bytes'],identity=v['identity'],sha256=v['sha256']) for k,v in sources.items()}
        from .host_budget import auxiliary_budget
        aux_budget=auxiliary_budget(extra,auxspec)
        event('auxiliary_memory_budget',**aux_budget)
        extra_after_aux=aux_budget['after_aux_bytes']
        aux=Arena(auxspec,admission_check=lambda rem:admit(rem+extra_after_aux),max_bytes=8*2**30)
        guard.track(aux);H.dontfork_arena(aux)
        if arm=='digit':event('numa_policy_installed',**bind_fresh(aux,'auxiliary'))
        aux.load(progress_callback(event,lambda:admit(aux.remaining_bytes+extra_after_aux),512*2**20));H.assert_dontfork(aux)
        hot=np.frombuffer(aux.mm,np.int64,count=inputs['cpu_cache_rows'],offset=aux.offsets['hot']);hot.flags.writeable=False
        if arm=='digit':inverse=np.frombuffer(aux.mm,np.int32,count=graph.storage_rows,offset=aux.offsets['inverse']);inverse.flags.writeable=False
        event('graph_registration_begin',bytes=arena.size)
        registration=Registration(arena)
        event('graph_registration_complete',bytes=arena.size)
        H.assert_dontfork(arena)
        event('native_sampler_begin')
        Native, _ = S.select(arm)
        native=Native(graph,registration)
        event('native_sampler_complete')
        admit(b['host_components']['cpu_features']+b['host_components']['backend_page_initialization']+b['host_components']['logical_slot_lookup']+16*2**30)
        provider=Provider(module,storage,arm,graph,hot,inverse,lambda:admit(16*2**30),event)
        numa_placement=dict(policy='unchanged',regions=[])
        if arm=='digit':
            check()
            numa_placement=dict(policy='interleave:0-1',regions=[placement(a) for a in (arena,aux,provider.external.pool)])
            event('numa_placement_verified',**numa_placement)
            check()
        engine=Engine(native,arm,model,opt,provider,lambda:admit(8*2**30))
        result=engine.window(roots,labels,P.WARMUP,P.MEASURED,event=lambda **kw:event(**kw))
        result.update(numa_placement=numa_placement,gpu_memory_admission=gpu_budget,io_probe=io_check,feature_value_checks=engine.feature_checks,gpu_torch_peak_bytes=torch.cuda.max_memory_allocated(),cache=provider.cache,arm=arm,mode=mode,manifest_sha256=digest,storage_acceptance_sha256=storage['acceptance_sha256'])
        result['execution_optimizations']=dict(reused_sampling_buffers=True,gpu_postprocessing=True)
        from .sampler_build import OUT as sampling_build
        import json
        result['sampling_optimization']=(dict(enabled=True,uncovered_rank='single_binary_search',eid_resolution='warp_cooperative',cpu_group_validation='compiled_one_scan_per_owner',
            cuda_sha256=json.loads((sampling_build/'libcompact_cuda.so.json').read_text())['sha256'],
            validator_sha256=json.loads((sampling_build/'libvalidate.so.json').read_text())['sha256']) if arm=='digit' else dict(enabled=False,baseline='v21'))
        # This is evidence of completed computation, never lifecycle acceptance.
        event('training_window_complete',measurement=result,lifecycle_complete=False)
    except BaseException as error:
        failure=record_failure(error)
    finally:
        try:
            if provider is not None:
                provider.close()
                if result is not None:result['external_cache_released']=provider.external.closed
            # Never inspect locals() here: its frame snapshot retains hot/inverse
            # exported views even after these fast local variables are cleared.
            engine=provider=None;hot=inverse=None
            if native is not None:native.close()
            if registration is not None:registration.close()
            if graph is not None:graph.close()
            native=registration=graph=None;gc.collect()
            for a in (aux,arena):
                if a is not None:a.close();guard.confirm_released(a)
            guard.release()
        except BaseException as error:
            cleanup_failure=record_failure(error)
            cleanup_failed(event,failure,cleanup_failure)
    raise_failures(failure,cleanup_failure)
    if verify_manifest()!=digest:raise RuntimeError('Training source changed')
    result.update(phase_accounting=phase_evidence,graph_and_aux_released=True,ownership=guard.receipt(),process_exit_acceptance_required=True)
    return result
