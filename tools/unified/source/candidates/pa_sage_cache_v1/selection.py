"""Stable Top-K selection and independently verified complete-page baseline."""
import numpy as np

def top(scores,k):
    if not 0<k<=len(scores):raise ValueError('Invalid budget')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
    above=np.flatnonzero(scores>threshold)
    ties=np.flatnonzero(scores==threshold)[:k-len(above)]
    return np.sort(np.r_[above,ties]).astype(np.int64),threshold

def page_rows(scores,k):
    if k%8:raise ValueError('CPU budget must use complete pages')
    pages,threshold=top(scores[:len(scores)//8*8].reshape(-1,8).sum(1),k//8)
    return (pages[:,None]*8+np.arange(8)).reshape(-1),threshold

def storage_rows_for_writer(rows):
    # Bound unused tail padding below 64 MiB, enabling exact 16384-page writes.
    quantum=8*16384
    return ((rows+quantum-1)//quantum)*quantum
