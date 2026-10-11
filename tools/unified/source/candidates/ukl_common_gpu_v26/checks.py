"""Small ordinary arrays only; no full graph or device access."""
import numpy as np
from candidates.ukl_native_sampling_v10.test_sampling import fixture

def tail_fixture():
    n=4096;origin=2**40+13
    idx=np.concatenate((np.ones(8192,np.int32),np.arange(2,1024,dtype=np.int32)))
    positions=np.arange(8192,len(idx)-1,3,dtype=np.int64)
    groups=len(positions)
    ptr=np.full(n+1,origin+len(idx),np.int64);ptr[0]=origin
    gp=np.full(n+1,groups,np.int64);gp[0]=0
    order=np.random.RandomState(9).permutation(groups)
    a=dict(ptr=ptr,idx=idx,gptr=gp,gidx=np.argsort(order).astype(np.int32),
           members=np.column_stack((idx[positions],idx[positions+1]))[order],
           covered=(np.column_stack((positions,positions+1)).ravel()+origin).astype(np.int64),
           bases=np.arange(groups,dtype=np.int64)*2+2**33+n,
           primary=np.arange(n,dtype=np.int64)+2**33)
    m=dict(nodes=n,edges=len(idx),groups=groups,units=groups,storage_rows=int(a['bases'][-1])+2,origin=origin,unit_origin=0)
    return a,m

def fixtures():
    return [fixture(),fixture(high=True),tail_fixture()]
