"""Deterministic logical-node rankings; sparse reverse PageRank and dry-run counts."""
import numpy as np
from .common import require, digest


def topk(scores, k):
    scores = np.asarray(scores)
    require(scores.ndim == 1 and np.issubdtype(scores.dtype, np.number), 'Expected numeric score vector')
    require(type(k) is int and 0 <= k <= len(scores), 'Invalid Top-K budget')
    require(np.isfinite(scores).all() and np.all(scores >= 0), 'Invalid ranking score')
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == len(scores):
        return np.arange(k, dtype=np.int64)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    ties = np.flatnonzero(scores == threshold)[:k - len(above)]
    return np.sort(np.concatenate((above, ties))).astype(np.int64)


def csc_check(indptr, indices):
    ptr, idx = np.asarray(indptr), np.asarray(indices)
    require(ptr.ndim == idx.ndim == 1 and ptr.dtype == idx.dtype == np.int64, 'CSC must use int64 vectors')
    require(len(ptr) >= 2 and ptr[0] == 0 and ptr[-1] == len(idx) and np.all(np.diff(ptr) >= 0), 'Invalid CSC pointers')
    require(not len(idx) or (idx.min() >= 0 and idx.max() < len(ptr) - 1), 'Invalid CSC node ID')
    return ptr, idx


def degree_scores(indptr, indices):
    ptr, _ = csc_check(indptr, indices)
    return np.diff(ptr).copy()


def reverse_pagerank(indptr, indices, damping=.85, iterations=20, chunk_edges=262144):
    """CSC holds src -> dst. Walk dst -> src, with uniform dangling mass."""
    ptr, idx = csc_check(indptr, indices)
    require(0 < damping < 1 and type(iterations) is int and iterations > 0 and
            type(chunk_edges) is int and chunk_edges > 0, 'Invalid PageRank parameters')
    n = len(ptr) - 1
    degree = np.diff(ptr)
    scores = np.full(n, 1. / n, dtype=np.float64)
    for _ in range(iterations):
        base = ((1 - damping) + damping * scores[degree == 0].sum()) / n
        nxt = np.full(n, base, dtype=np.float64)
        for lo in range(0, len(idx), chunk_edges):
            hi = min(len(idx), lo + chunk_edges)
            owners = np.searchsorted(ptr, np.arange(lo, hi, dtype=np.int64), side='right') - 1
            np.add.at(nxt, idx[lo:hi], damping * scores[owners] / degree[owners])
        require(np.isfinite(nxt).all() and np.all(nxt >= 0) and abs(nxt.sum() - 1.) < 1e-8, 'PageRank lost mass')
        scores = nxt
    return scores


class FrequencyProfile:
    def __init__(self, nodes, seed, purpose='independent_presampling'):
        require(type(nodes) is int and nodes > 0 and purpose == 'independent_presampling',
                'Only a separate presampling phase may select frequency hot nodes')
        self.counts = np.zeros(nodes, dtype=np.int64)
        self.seed, self.batches, self.total = seed, 0, 0
        self.sealed = False

    def observe(self, logical_inputs):
        require(not self.sealed, 'Frequency profile is already frozen')
        ids = np.asarray(logical_inputs)
        require(ids.ndim == 1 and ids.dtype == np.int64, 'Input IDs must be int64')
        require(not len(ids) or (ids.min() >= 0 and ids.max() < len(self.counts)), 'Input node out of range')
        require(len(np.unique(ids)) == len(ids), 'Count feature input rows, not duplicate edge occurrences')
        require(not len(ids) or self.counts[ids].max() < np.iinfo(np.int64).max, 'Counter overflow')
        self.counts[ids] += 1
        self.batches += 1
        self.total += len(ids)

    def freeze(self):
        require(self.batches > 0, 'Empty profile')
        self.sealed = True
        self.counts.setflags(write=False)
        return dict(purpose='independent_presampling', seed=self.seed, batches=self.batches,
                    total_logical_requests=self.total, counts_sha256=digest(self.counts),
                    optimizer_updates=0, evaluation_calls=0)
