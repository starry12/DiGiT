"""Derive physical g2 heads without changing any sampled nodes or edges."""
import numpy as np


def group_bases(graph, rows):
    rows = np.asarray(rows)
    n, count = graph.nodes, graph.storage_rows
    if rows.dtype != np.int64 or rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= count):
        raise ValueError('Invalid physical row vector')
    if (count-n) % 2:
        raise ValueError('Invalid packed replica extent')
    replicas = (count-n)//2
    primary = graph.groups-replicas
    if not 0 <= 2*primary <= n:
        raise ValueError('Invalid primary group count')
    heads = np.full(len(rows), -1, np.int64)
    p = rows < 2*primary
    r = rows >= n
    heads[p] = (rows[p]//2)*2
    # N is odd for UKL: replica heads are odd. XOR 1 would select a wrong row.
    heads[r] = n+((rows[r]-n)//2)*2
    grouped = p | r
    ids = np.where(p[grouped], heads[grouped]//2,
                   primary+(heads[grouped]-n)//2)
    if not np.array_equal(graph.arrays['bases'][ids], heads[grouped]):
        raise ValueError('Physical group heads differ from accepted layout')
    return heads
