"""Chunked first-CSC-occurrence EIDs, matching resolve_eids_kernel exactly."""
import numpy as np


def resolve_first(indptr, indices, eids, seeds, sources, chunk=65536):
    seeds=np.asarray(seeds,dtype=np.int64)
    sources=np.asarray(sources,dtype=np.int64)
    if sources.ndim!=2 or sources.shape[0]!=len(seeds) or chunk<=0:
        raise ValueError('invalid sampled slot matrix')
    result=np.full(sources.shape,-1,dtype=np.int64)
    for row,owner in enumerate(seeds):
        if owner<0 or owner+1>=len(indptr):raise ValueError('invalid seed')
        wanted=set(int(x) for x in sources[row] if x>=0)
        found={}
        start,end=int(indptr[owner]),int(indptr[owner+1])
        for offset in range(start,end,chunk):
            if not wanted:break
            values=np.asarray(indices[offset:min(end,offset+chunk)])
            for node in tuple(wanted):
                positions=np.flatnonzero(values==node)
                if positions.size:
                    found[node]=int(eids[offset+int(positions[0])]);wanted.remove(node)
        for col,node in enumerate(sources[row]):
            if node>=0:result[row,col]=found.get(int(node),-2)
    return result
