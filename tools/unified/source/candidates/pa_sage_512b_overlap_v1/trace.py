"""Measure intersections of actual CUDA read and model kernels, not host spans."""
import json
import re
from pathlib import Path


def overlap_evidence(path):
    trace=json.loads(Path(path).read_text())
    kernels=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='kernel' and e.get('dur',0)>0]
    reads=[e for e in kernels if 'read_feature_kernel' in e.get('name','')]
    # Conservative whitelist excludes sampling, address resolution, memcpy and WB.
    model=[e for e in kernels if re.search(r'gemm|[sS]p[Mm][Mm]|GSpMM|spmm',e.get('name',''))]
    intersections=[]
    for a in reads:
        for b in model:
            lo=max(a['ts'],b['ts']);hi=min(a['ts']+a['dur'],b['ts']+b['dur'])
            aa=a.get('args',{});bb=b.get('args',{})
            streams_known=aa.get('stream') is not None and bb.get('stream') is not None
            same_device=aa.get('device') is not None and aa.get('device')==bb.get('device')
            if hi>lo and streams_known and same_device and aa['stream']!=bb['stream']:
                intersections.append((lo,hi))
    merged=[]
    for lo,hi in sorted(intersections):
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
        else:merged.append([lo,hi])
    return dict(trace_path=str(path),read_kernels=len(reads),model_kernels=len(model),
        read_kernel_names=sorted({e['name'] for e in reads}),model_kernel_names=sorted({e['name'] for e in model}),
        overlap_intersections=len(intersections),overlap_union_ms=sum(hi-lo for lo,hi in merged)/1000,
        real_gpu_overlap_observed=bool(intersections),scope='16 profiled steps; excluded from performance means',
        qualification='Only actual read kernels intersecting known GEMM/SpMM kernels on other streams count')
