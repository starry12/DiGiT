"""Logical CPU cache aliases; independent of native backend and GPU policy."""
import numpy as np
def normalize_hot_requests(hot,primary,logical,physical,flags):
    pos=np.searchsorted(hot,logical);mask=pos<len(hot)
    mask[mask]=hot[pos[mask]]==logical[mask]
    rows=np.array(physical,copy=True);group=np.array(flags,copy=True)
    rows[mask]=primary[logical[mask]];group[mask]=False
    return rows,group
