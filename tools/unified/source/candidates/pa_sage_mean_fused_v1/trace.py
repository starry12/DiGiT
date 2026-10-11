"""Confirm exact normalization kernel counts in a separate 16-step trace."""
from .common import read,require,sha


def kernel_evidence(path,mode):
    require(mode in ('legacy','fused'),'Unknown trace mode')
    events=read(path)['traceEvents']
    kernels=[e for e in events if e.get('ph')=='X' and e.get('cat')=='kernel']
    mean=[e for e in kernels if 'mean_normalize_kernel<' in e.get('name','')]
    forward=[e for e in mean if ', false>' in e['name'] or ', (bool)0>' in e['name']]
    backward=[e for e in mean if ', true>' in e['name'] or ', (bool)1>' in e['name']]
    require(len(mean)==len(forward)+len(backward),'Unrecognized mean kernel name')
    require(len(forward)==(48 if mode=='fused' else 0),'Missing three mean forwards per step')
    require(len(backward)==(32 if mode=='fused' else 0),'Missing two mean backwards per step')
    group=[e for e in kernels if 'group_sample_' in e.get('name','')]
    eid=[e for e in kernels if 'resolve_eids_kernel' in e.get('name','')]
    graph=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cuda_runtime' and e.get('name','').startswith('cudaGraphLaunch')]
    require(len(group)==len(eid)==16 and all('incremental' in e['name'] for e in group),'Sampler changed')
    require(len(graph)==32,'Dense graph replay changed')
    cpu={name:dict(calls=sum(e.get('ph')=='X' and e.get('cat')=='cpu_op' and e.get('name')==name for e in events),
        inclusive_host_ms=sum(e.get('dur',0) for e in events if e.get('ph')=='X' and e.get('cat')=='cpu_op' and e.get('name')==name)/1000)
        for name in ('aten::clamp','aten::_to_copy','aten::div','GSpMM')}
    return dict(mode=mode,trace_sha256=sha(path),mean_forward_calls=len(forward),mean_backward_calls=len(backward),
        mean_forward_ms=sum(e['dur'] for e in forward)/1000,mean_backward_ms=sum(e['dur'] for e in backward)/1000,
        group_calls=len(group),eid_calls=len(eid),graph_launches=len(graph),cpu_ops=cpu,
        scope='16 diagnostic full batches; excluded from performance',performance_evidence=False)


def compare_traces(legacy,fused):
    require(legacy['mode']=='legacy' and fused['mode']=='fused','Wrong trace modes')
    require(legacy['graph_launches']==fused['graph_launches']==32,'Dense capture changed')
    return dict(legacy=legacy,fused=fused,
        limits='Inclusive CPU durations overlap; short profiled windows cannot predict full epoch speedup')
