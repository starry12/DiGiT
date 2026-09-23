"""Retain feature layout/groups; add reverse raw neighbors, with exact validation."""
import copy
import numpy as np
from training.sage.common import require

def rewrite(oldptr,oldidx,originalptr,bptr,bidx,beid,edge_count,rp,ri,notify=lambda **kw:None,chunk=100000):
    n=len(bptr)-1
    require(len(originalptr)==n+1,'Wrong source CSC')
    rp[:n+1]=oldptr[:n+1]+(bptr-originalptr)
    rp[n+1:]=rp[n]
    require(int(rp[-1])==len(ri)==len(oldidx)+edge_count,'Wrong bidirectional unit count')
    for lo in range(0,n,chunk):
        hi=min(n,lo+chunk);a,b=int(bptr[lo]),int(bptr[hi]);x,y=int(oldptr[lo]),int(oldptr[hi])
        mask=(beid[a:b]>=edge_count)&(beid[a:b]<2*edge_count)
        reverse_counts=np.add.reduceat(mask.astype(np.int64),bptr[lo:hi]-a)
        old_counts=np.diff(oldptr[lo:hi+1])
        require(np.array_equal(reverse_counts,np.diff(bptr[lo:hi+1])-np.diff(originalptr[lo:hi+1])),'Reverse-edge degree mismatch')
        dest=np.arange(x,y)+np.repeat(rp[lo:hi]-oldptr[lo:hi],old_counts)
        ri[dest]=oldidx[x:y]
        prefix=np.r_[0,np.cumsum(reverse_counts)]
        dest=np.arange(int(prefix[-1]))+np.repeat(rp[lo:hi]+old_counts-prefix[:-1],reverse_counts)
        ri[dest]=bidx[a:b][mask]
        if lo%5000000==0:notify(stage='rewrite',nodes_done=hi,total=n)
    return rp,ri

def exact_validate(rp,ri,bptr,bidx,members,chunk=100000,notify=lambda **kw:None):
    n=len(bptr)-1;g=members.shape[1];seen=0
    require(np.all(np.diff(rp)>=0) and int(rp[-1])==len(ri),'Invalid overlay pointer')
    for lo in range(0,n,chunk):
        hi=min(n,lo+chunk);units=ri[int(rp[lo]):int(rp[hi])];owners=np.repeat(np.arange(lo,hi,dtype=np.int64),np.diff(rp[lo:hi+1]))
        group=units>=n
        require(not len(units) or (units.min()>=0 and units.max()<n+len(members)),'Invalid compact node ID')
        actual=np.concatenate((owners[~group]*n+units[~group],(owners[group,None]*n+members[units[group]-n]).reshape(-1)))
        expected=np.repeat(np.arange(lo,hi,dtype=np.int64),np.diff(bptr[lo:hi+1]))*n+bidx[int(bptr[lo]):int(bptr[hi])]
        actual.sort();expected.sort()
        require(np.array_equal(actual,expected),'Expanded adjacency multiset differs at node '+str(lo));seen+=int(group.sum())
        if lo%5000000==0:notify(stage='exact_adjacency',nodes_done=hi,total=n)
    require(seen==len(members),'Group count changed')
    return dict(passed=True,nodes=n,edges=len(bidx),units=len(ri),groups=seen,preserves_parallel_edges=True)

def apply(base,ptr,idx,edge_count):
    from digit.artifacts import ArtifactBundle
    m=copy.deepcopy(base.manifest);arrays=dict(base.arrays)
    m['dataset']['num_edges']=int(edge_count)
    for name,a in [('reordered_indptr',ptr),('reordered_indices',idx)]:
        arrays[name]=a;m['files'][name]['shape']=list(a.shape)
    # This is an in-memory adapter, never represented as a newly verified disk
    # artifact. Its provenance is the separately hashed overlay prepared.json.
    m.setdefault('metadata',{})['bidirectional_overlay']=True
    return ArtifactBundle(base.root,m,arrays)
