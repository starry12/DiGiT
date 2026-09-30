"""Disk-backed Degree / RevPR on the already verified symmetric training graph."""
import time
import numpy as np
from candidates.pa_sage_cache_policy_v1.selection import topk
from .common import read, require, sha, write, write_new, identity, heavy_gate, verify, progress


def owner_chunks(ptr, max_nodes=65536, max_edges=1048576):
    n = len(ptr)-1; lo = 0
    while lo < n:
        hi = min(n, lo+max_nodes)
        if ptr[hi]-ptr[lo] > max_edges:
            hi = max(lo+1, min(hi, int(np.searchsorted(ptr, ptr[lo]+max_edges, side='right'))-1))
        yield lo, hi
        lo = hi


def symmetric_revpr(ptr, idx, directory, damping=.85, iterations=20, notify=lambda **kw: None):
    """For the verified bidirectional graph, reverse walk equals forward walk.

    Sum incoming rank/degree with segmented reductions; parallel edges and
    loops retain multiplicity. Avoid an edge-by-edge random np.add.at pass.
    Uses four disk-backed vectors and bounded edge chunks, never a dense matrix.
    """
    from pathlib import Path
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    n = len(ptr)-1
    require(n > 0 and ptr[0] == 0 and ptr[-1] == len(idx) and
            0 < damping < 1 and type(iterations) is int and iterations > 0, 'Bad PageRank geometry')
    degree = np.lib.format.open_memmap(directory/'degree_scores.npy', mode='w+', dtype=np.int64, shape=(n,))
    scores = np.lib.format.open_memmap(directory/'rank_a.npy', mode='w+', dtype=np.float64, shape=(n,))
    nxt = np.lib.format.open_memmap(directory/'rank_b.npy', mode='w+', dtype=np.float64, shape=(n,))
    weights = np.lib.format.open_memmap(directory/'weights.npy', mode='w+', dtype=np.float64, shape=(n,))
    for lo, hi in owner_chunks(ptr):
        degree[lo:hi] = np.diff(ptr[lo:hi+1])
        require(np.all(degree[lo:hi] >= 0), 'Decreasing CSC pointers')
        rows = idx[int(ptr[lo]):int(ptr[hi])]
        require(not len(rows) or (rows.min() >= 0 and rows.max() < n), 'Invalid graph node')
        scores[lo:hi] = 1./n
    degree.flush(); scores.flush()
    for iteration in range(iterations):
        dangling = 0.
        for lo in range(0, n, 1048576):
            hi = min(n, lo+1048576); d = degree[lo:hi]; x = scores[lo:hi]
            dangling += float(x[d == 0].sum())
            weights[lo:hi] = np.divide(x, d, out=np.zeros_like(x), where=d != 0)
        base = ((1-damping)+damping*dangling)/n
        last = time.monotonic()
        for lo, hi in owner_chunks(ptr):
            start, end = int(ptr[lo]), int(ptr[hi]); valid = degree[lo:hi] > 0
            values = np.full(hi-lo, base, dtype=np.float64)
            if end > start:
                offsets = (ptr[lo:hi]-start)[valid]
                values[valid] += damping*np.add.reduceat(weights[idx[start:end]], offsets)
            nxt[lo:hi] = values
            if time.monotonic()-last >= 10:
                notify(iteration=iteration+1, nodes_done=hi, nodes=n); last=time.monotonic()
        require(np.isfinite(nxt).all() and np.all(nxt >= 0) and abs(float(nxt.sum())-1.) < 1e-8, 'PageRank lost mass')
        nxt.flush(); scores, nxt = nxt, scores
        notify(iteration=iteration+1, nodes_done=n, nodes=n)
    return degree, scores


def select_to_file(scores, count, path):
    hot = topk(scores, count)
    with path.open('xb') as stream: np.save(stream, hot, allow_pickle=False)
    return dict(path=str(path), sha256=sha(path), identity=identity(path), rows=len(hot))


def execute(p, protocol_path, binding_path, output, large):
    heavy_gate(); execution = verify()
    from .binding import check
    binding=read(binding_path); check(binding, protocol_path, execution)
    require(p['graph']['arm'] == 'bidirectional', 'Segmented RevPR requires the bound symmetric training graph')
    from .common import ROOT
    data=ROOT/p['data']; ptr=np.load(data/'original_indptr.npy',mmap_mode='r'); idx=np.load(data/'original_indices.npy',mmap_mode='r')
    started=time.time()
    degree, revpr=symmetric_revpr(ptr, idx, large, **{k:p['ranking']['revpr'][k] for k in ('damping','iterations')},
        notify=lambda **kw:progress(output,'ranking',**kw))
    records={name:select_to_file(values,p['arms'][name]['cpu_rows'],large/(name+'_hot.npy'))
             for name,values in [('degree',degree),('revpr',revpr)]}
    check(binding,protocol_path,execution)
    report=dict(kind='cache_policy_graph_rankings',passed=True,fixture=False,source_sha256=execution,
        protocol_sha256=sha(protocol_path),binding_sha256=sha(binding_path),graph_sha256=binding['graph_sha256'],
        hot=records,ranking=p['ranking'],seconds=time.time()-started,raw_ssd_access=False,gpu_used=False,
        revpr_score_path=str(revpr.filename),degree_score_path=str(degree.filename))
    write(output/'report.json',report)
