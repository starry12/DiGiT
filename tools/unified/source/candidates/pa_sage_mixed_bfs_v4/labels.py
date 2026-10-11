"""Lossless class storage; validate every official training target before upload."""
import hashlib
import numpy as np


def pack_labels(source,train,classes,chunk=1048576):
    if not 0<classes<255 or chunk<=0:raise ValueError('Invalid compact class geometry')
    source=np.asarray(source).reshape(-1);train=np.asarray(train)
    if train.ndim!=1 or not np.issubdtype(train.dtype,np.integer) or (train.size and (train.min()<0 or train.max()>=len(source))):
        raise ValueError('Invalid training IDs')
    packed=np.full(len(source),255,dtype=np.uint8);known=0
    for lo in range(0,len(source),chunk):
        values=source[lo:lo+chunk];finite=np.isfinite(values);valid=values[finite]
        if np.any(~finite & ~np.isnan(values)) or np.any(valid<0) or np.any(valid>=classes) or np.any(valid!=np.floor(valid)):
            raise ValueError('Label cannot be represented as a valid class')
        packed[lo:lo+chunk][finite]=valid.astype(np.uint8);known+=int(finite.sum())
    targets=packed[train]
    if np.any(targets==255):raise ValueError('Missing label in official training split')
    reference=np.asarray(source[train],dtype=np.int64)
    if not np.array_equal(targets.astype(np.int64),reference):raise RuntimeError('Training labels changed')
    return packed,dict(passed=True,dtype='uint8',missing_sentinel=255,classes=classes,
        nodes=len(source),known_labels=known,train_examples=len(train),training_targets_exact=True,
        gpu_storage_bytes=packed.nbytes,previous_gpu_storage_bytes=len(source)*8,
        saved_gpu_bytes=len(source)*7,train_target_sha256=hashlib.sha256(reference.tobytes()).hexdigest())


def upload_labels(source,train,classes,device):
    import torch
    packed,audit=pack_labels(source,train,classes)
    labels=torch.from_numpy(packed).to(device=device)
    if labels.dtype!=torch.uint8 or labels.numel()!=len(source):raise RuntimeError('Wrong GPU label storage')
    return labels,audit
