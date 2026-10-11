"""Checked, bounded host conversion; external IDs/offsets remain int64."""
import numpy as np
import torch

LIMIT=2**31-1


def check_ids(array,chunk=65536):
    if chunk<=0 or not np.issubdtype(array.dtype,np.integer):raise ValueError('integer IDs required')
    # Row-wise slicing preserves boundedness even for non-C-contiguous arrays.
    if array.ndim not in (1,2):raise ValueError('expected vector or member matrix')
    rows=max(1,chunk//(array.shape[1] if array.ndim==2 else 1))
    for start in range(0,len(array),rows):
        part=array[start:start+rows]
        if part.size and (part.min()<0 or part.max()>LIMIT):
            raise ValueError('ID outside nonnegative signed int32 range')


def upload_ids(array,device,chunk=65536):
    # Caller validates every array before allocating any compact metadata.
    target=torch.empty(array.shape,dtype=torch.int32,device=device)
    rows=max(1,chunk//(array.shape[1] if array.ndim==2 else 1))
    for start in range(0,len(array),rows):
        part=np.array(array[start:start+rows],dtype=np.int32,copy=True)
        target[start:start+rows].copy_(torch.from_numpy(part))
    return target
