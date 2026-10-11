"""Checked, bounded host conversion; external IDs/offsets remain int64."""
import numpy as np
import torch

LIMIT=2**31-1


def check_ids(array,chunk=65536):
    if chunk<=0 or not np.issubdtype(array.dtype,np.integer):raise ValueError('integer IDs required')
    # Row-wise slicing preserves boundedness even for non-C-contiguous arrays.
    if array.ndim not in (1,2):raise ValueError('expected vector or member matrix')
    rows=max(1,chunk//(max(1,array.shape[1]) if array.ndim==2 else 1))
    for start in range(0,len(array),rows):
        part=array[start:start+rows]
        if part.size and (part.min()<0 or part.max()>LIMIT):
            raise ValueError('ID outside nonnegative signed int32 range')


def upload_ids(array,device,chunk=4194304):
    # Caller validates every array before allocating any compact metadata.
    if chunk <= 0 or array.ndim not in (1, 2):
        raise ValueError('positive chunk and vector/member matrix required')
    target=torch.empty(array.shape,dtype=torch.int32,device=device)
    width=array.shape[1] if array.ndim==2 else 1
    rows=max(1,chunk//max(1,width))
    if not array.size:
        return target
    # One bounded staging allocation, reused only after each blocking copy returns.
    shape=(min(rows,len(array)),width) if array.ndim==2 else (min(rows,len(array)),)
    staging=torch.empty(shape,dtype=torch.int32,pin_memory=target.is_cuda)
    host=staging.numpy()
    for start in range(0,len(array),rows):
        count=min(rows,len(array)-start)
        np.copyto(host[:count],array[start:start+count],casting='unsafe')
        target[start:start+count].copy_(staging[:count])
    return target


def check_all_ids(arrays):
    # Finish every range check before the caller allocates compact GPU metadata.
    from concurrent.futures import ThreadPoolExecutor
    arrays=list(arrays)
    if not arrays:
        return
    with ThreadPoolExecutor(max_workers=min(4,len(arrays))) as pool:
        list(pool.map(check_ids,arrays))
