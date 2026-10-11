"""Setup-only mapping. Training uses the existing native page metadata."""
import numpy as np
from candidates.pa_sage_cache_policy_v1.mapping import storage_slots


def alias_chunks(physical_hot, inverse, nodes, chunk_rows=1048576):
    physical_hot = np.asarray(physical_hot)
    if (physical_hot.dtype != np.int64 or physical_hot.ndim != 1 or not len(physical_hot)
            or not np.all(physical_hot[1:] > physical_hot[:-1])
            or physical_hot[0] < 0 or physical_hot[-1] >= len(inverse)
            or len(physical_hot) >= 2**31 or not 0 < chunk_rows <= 1048576):
        raise ValueError('Invalid primary cache rows/chunk size')
    logical = np.asarray(inverse[physical_hot], dtype=np.int64)
    if np.any((logical < 0) | (logical >= nodes)) or len(np.unique(logical)) != len(logical):
        raise ValueError('Primary rows must represent unique valid logical nodes')
    slots = np.zeros(nodes, dtype=np.uint32)
    # Cache is populated in physical address order, not logical node order.
    slots[logical] = np.arange(1, len(logical)+1, dtype=np.uint32)
    for lo in range(0, len(inverse), chunk_rows):
        yield lo, storage_slots(inverse[lo:lo+chunk_rows], slots)
