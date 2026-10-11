"""Small graph retaining the real failure's nine raw units plus one pair."""
from types import SimpleNamespace
import numpy as np


def fixture():
    n=16;units=list(range(9))+[n]
    f=dict(n=n,groups=1,group_size=2,ptr=[0]+[len(units)]*n,idx=units,
        members=[9,10],bases=[100],mapping=[0],primary=list(range(n)),
        optr=[0]+[11]*n,oidx=list(range(11)),eids=list(range(11)),seeds=[0])
    bundle=SimpleNamespace(manifest=dict(dataset=dict(num_nodes=n),grouping=dict(group_size=2)),
        arrays=dict(reordered_indptr=np.asarray(f['ptr']),reordered_indices=np.asarray(units),
            group_members=np.array([[9,10]]),supernode_to_group=np.array([0])))
    return f,bundle,(np.asarray(f['optr']),np.asarray(f['oidx']))


def block(sources,groups,count,n=16):
    import torch,dgl
    b=dgl.create_block((torch.tensor(sources,dtype=torch.int64),torch.zeros(len(sources),dtype=torch.int64)),num_src_nodes=n,num_dst_nodes=1)
    b.srcdata[dgl.NID]=torch.arange(n);b.dstdata[dgl.NID]=torch.tensor([0])
    b.edata[dgl.EID]=torch.tensor(sources,dtype=torch.int64)
    b.dstdata['digit_sampled_groups']=torch.tensor([groups])
    b.dstdata['digit_sampled_nodes']=torch.tensor([count])
    return b
