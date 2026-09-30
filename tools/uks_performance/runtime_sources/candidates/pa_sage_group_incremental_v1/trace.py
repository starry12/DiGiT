"""Compare group kernel times while checking unchanged EID and dense replay."""
from .common import read, require, sha


def kernel_evidence(path, mode):
    events=read(path)['traceEvents']
    kernels=[e for e in events if e.get('ph')=='X' and e.get('cat')=='kernel']
    group=[e for e in kernels if 'group_sample_' in e.get('name','')]
    eid=[e for e in kernels if 'resolve_eids_kernel' in e.get('name','')]
    graph=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cuda_runtime'
           and e.get('name','').startswith('cudaGraphLaunch')]
    require(len(group)==len(eid)==16,'Missing group/EID diagnostic window')
    require(len(graph)==32,'Both arms must replay the same dense forward/backward')
    changed=[e for e in group if 'incremental' in e['name']]
    require(len(changed)==(16 if mode=='incremental' else 0),'Wrong group kernel')
    return dict(mode=mode,trace_sha256=sha(path),group_calls=len(group),eid_calls=len(eid),
        graph_launches=len(graph),group_kernel_ms=sum(e['dur'] for e in group)/1000,
        eid_kernel_ms=sum(e['dur'] for e in eid)/1000,group_kernel_names=sorted({e['name'] for e in group}),
        scope='16 diagnostic full batches; excluded from performance means',performance_evidence=False)


def compare_traces(legacy, incremental):
    require(legacy['mode']=='legacy' and incremental['mode']=='incremental','Wrong trace modes')
    require(legacy['group_calls']==incremental['group_calls']==16,'Incomplete group trace')
    require(legacy['graph_launches']==incremental['graph_launches']==32,'Dense graph changed')
    return dict(legacy=legacy,incremental=incremental,
        group_kernel_reduction_percent=100*(1-incremental['group_kernel_ms']/legacy['group_kernel_ms']),
        limits='A small profiled kernel window cannot predict full-epoch E2E speedup')
