"""Bounded CUDA pairing/scatter check; no host pinning, graph, or raw SSD."""
import numpy as np


def run(module):
    import torch
    cases=[([0,1],[0,0],[],[1,0]),
           ([1,2],[1,1],[],[1,0]),
           ([7,8],[7,7],[],[1,0]),
           ([1,2],[1,1],[1],[-1,-1]),
           ([1],[1],[],[-1]),
           ([1,2],[-1,-1],[],[-1,-1]),
           ([1,2],[1,2],[],[-1,-1]),
           ([2,1],[1,1],[],[1,0]),
           ([0,1,3,4],[0,0,3,3],[],[1,0,3,2])]
    for ids,heads,hot,wanted in cases:
        i=torch.tensor(ids,dtype=torch.int64,device='cuda:0')
        h=torch.tensor(heads,dtype=torch.int64,device='cuda:0')
        c=torch.zeros(16,dtype=torch.int32,device='cuda:0')
        if hot:c[hot]=1
        table=torch.empty(32,dtype=torch.int32,device='cuda:0')
        partners=torch.empty(len(ids),dtype=torch.int32,device='cuda:0')
        errors=torch.zeros(1,dtype=torch.int64,device='cuda:0')
        module.probe_pair_hash(i.data_ptr(),h.data_ptr(),c.data_ptr(),len(ids),16,table.data_ptr(),32,partners.data_ptr(),errors.data_ptr())
        if partners.cpu().tolist()!=wanted or errors.item():
            raise RuntimeError('Native group-head pairing differs from CPU reference')
    src=torch.arange(256,dtype=torch.float32,device='cuda:0')
    a=torch.zeros(128,dtype=torch.float32,device='cuda:0');b=torch.zeros_like(a)
    module.probe_pair_scatter(src.data_ptr(),a.data_ptr(),b.data_ptr())
    if not torch.equal(a,src[:128]) or not torch.equal(b,src[128:]):
        raise RuntimeError('512B dual-slot scatter mismatch')
    return dict(passed=True,pair_cases=len(cases),scatter_checked=True,
                graph_registered=False,raw_ssd_access=False)


if __name__=='__main__':
    import importlib.util,json
    from .build import BINARY
    from .backend import verify_binary
    verify_binary()
    spec=importlib.util.spec_from_file_location('BAM_Feature_Store',str(BINARY))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    print(json.dumps(run(module)))
