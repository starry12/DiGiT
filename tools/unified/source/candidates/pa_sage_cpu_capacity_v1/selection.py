"""Nested exact row sets from one immutable frequency vector; never per-arm profiles."""
import numpy as np
from .common import ARMS, require, sha, identity
from candidates.pa_sage_cache_policy_v1.selection import topk


def nested(hot, p, chunk_rows=1048576):
    previous=np.empty(0,np.int64)
    for arm in ARMS:
        ids=hot[arm]
        require(ids.ndim==1 and ids.dtype==np.int64 and len(ids)==p['arms'][arm]['cpu_rows'], 'Wrong hot-set extent')
        last=-1
        for lo in range(0,len(ids),chunk_rows):
            block=ids[lo:lo+chunk_rows]
            require(block[0]>last and block[-1]<p['graph']['nodes'] and np.all(np.diff(block)>0), 'Hot IDs invalid or unsorted')
            last=int(block[-1])
        for lo in range(0,len(previous),chunk_rows):
            block=previous[lo:lo+chunk_rows];positions=np.searchsorted(ids,block)
            require(np.all(positions<len(ids)) and np.array_equal(ids[positions],block), 'Hot sets are not nested')
        previous=ids
    return True


def select_all(counts,p,large):
    require(counts.dtype==np.int64 and counts.ndim==1 and len(counts)==p['graph']['nodes'], 'Wrong frequency vector')
    hot={};arrays={}
    for arm in ARMS:
        path=large/(arm+'_hot.npy');require(not path.exists(),'Preserve previous selection')
        np.save(path,topk(counts,p['arms'][arm]['cpu_rows']),allow_pickle=False)
        arrays[arm]=np.load(path,mmap_mode='r')
        hot[arm]=dict(path=str(path),sha256=sha(path),identity=identity(path),rows=len(arrays[arm]))
    nested(arrays,p)
    return hot
