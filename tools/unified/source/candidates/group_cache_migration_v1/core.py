"""Full training-epoch profiles and counterfactual static CPU-cache audits.

Only logical, deduplicated feature inputs are counted, before any cache filter.
Audits are CPU eligibility, not actual CPU/GPU/SSD serving or training accuracy.
"""
import hashlib
import math
import time
import numpy as np
from .common import require

BATCH = 1024
PROFILE_SEED = 23
AUDIT_SEEDS = (0, 1)
CHUNK = 1048576


def digest_array(array):
    h = hashlib.sha256()
    for lo in range(0, len(array), CHUNK):
        h.update(np.ascontiguousarray(array[lo:lo+CHUNK]).tobytes())
    return h.hexdigest()


def batches(canonical, epoch=0, batch_size=BATCH):
    """Preserve BFS minibatches; shuffle full blocks; include final partial block."""
    complete = len(canonical)//batch_size
    order = np.random.default_rng(np.random.SeedSequence([0, epoch])).permutation(complete)
    for index in order:
        yield canonical[int(index)*batch_size:(int(index)+1)*batch_size]
    if len(canonical) % batch_size:
        yield canonical[complete*batch_size:]


def validate_order(canonical, training, nodes):
    require(canonical.dtype == np.int64 and training.dtype == np.int64,
            'Training order must be int64')
    require(canonical.ndim == training.ndim == 1 and len(canonical) == len(training) > 0,
            'Full training split required')
    require(int(canonical.min()) >= 0 and int(canonical.max()) < nodes, 'Invalid root ID')
    # An exact permutation check; bounded by the training split (10% of graph).
    ordered = np.sort(canonical)
    require(np.array_equal(ordered, training) and np.all(ordered[1:] > ordered[:-1]),
            'BFS must cover every training root exactly once')


def logical_ids(ids, nodes):
    ids = np.asarray(ids)
    require(ids.ndim == 1 and ids.dtype.kind in 'iu' and len(ids) > 0,
            'Missing logical input nodes')
    require(int(ids.min()) >= 0 and int(ids.max()) < nodes, 'Logical ID out of range')
    require(len(np.unique(ids)) == len(ids), 'Duplicate logical input nodes in batch')
    return ids


def profile(canonical, nodes, sampler_factory, event=lambda **kw: None, check=lambda: None, batch_iterator=None):
    started = time.perf_counter()
    counts = np.zeros(nodes, np.uint32)
    sample = sampler_factory(PROFILE_SEED)
    total = roots_seen = 0
    roots_hash = hashlib.sha256()
    count_batches = math.ceil(len(canonical)/BATCH)
    for i, roots in enumerate(batches(canonical) if batch_iterator is None else batch_iterator):
        check()
        ids = logical_ids(sample(roots, i), nodes)
        require(not np.any(counts[ids] == np.iinfo(np.uint32).max), 'Frequency overflow')
        counts[ids] += 1
        total += len(ids); roots_seen += len(roots); roots_hash.update(roots.tobytes())
        if i % 100 == 0 or i+1 == count_batches:
            event(stage='full_epoch_profile', batch=i+1, batches=count_batches,
                  roots_seen=roots_seen, logical_requests=total,
                  elapsed_seconds=time.perf_counter()-started)
    require(roots_seen == len(canonical), 'Incomplete profiling epoch')
    histogram = np.zeros(count_batches+1, np.int64)
    for lo in range(0, nodes, CHUNK):
        check()
        histogram += np.bincount(counts[lo:lo+CHUNK].astype(np.int64), minlength=len(histogram))
    observed = int(nodes-histogram[0])
    receipt = dict(complete=True, epochs=1, seed=PROFILE_SEED, batches=count_batches,
                   roots=roots_seen, roots_sha256=roots_hash.hexdigest(),
                   canonical_sha256=digest_array(canonical), logical_requests=total,
                   observed_nodes=observed, observed_node_fraction=observed/nodes,
                   counts_sha256=digest_array(counts), histogram=histogram.tolist(),
                   seconds=time.perf_counter()-started, feature_reads=0, optimizer_updates=0,
                   validation_calls=0, test_calls=0, profile_seed_is_independent=True)
    return counts, receipt


def select_hot(counts, capacity, revpr_hot, revpr_scores=None, check=lambda: None):
    """Frequency top-K; zero-count vacancies use ranked RevPR top-K candidates.

    Positive-count ties use logical ID. RevPR scores are required only for the
    fallback and correspond to revpr_hot, not to the entire graph.
    """
    nodes = len(counts)
    require(0 < capacity <= nodes and len(revpr_hot) == capacity, 'Cache capacity mismatch')
    require(np.all(revpr_hot[1:] > revpr_hot[:-1]) and revpr_hot[0] >= 0 and revpr_hot[-1] < nodes,
            'Invalid logical RevPR set')
    max_count = int(counts.max())
    hist = np.zeros(max_count+1, np.int64)
    for lo in range(0, nodes, CHUNK):
        check(); hist += np.bincount(counts[lo:lo+CHUNK].astype(np.int64), minlength=len(hist))
    observed = nodes-int(hist[0])
    keep = min(capacity, observed)
    threshold = 1
    if keep:
        threshold = max_count-int(np.searchsorted(np.cumsum(hist[:0:-1]), keep))
    hot = np.empty(capacity, np.int64); cursor = 0
    for lo in range(0, nodes, CHUNK):
        check(); ids = np.flatnonzero(counts[lo:lo+CHUNK] > threshold)+lo
        hot[cursor:cursor+len(ids)] = ids; cursor += len(ids)
    for lo in range(0, nodes, CHUNK):
        check(); ids = (np.flatnonzero(counts[lo:lo+CHUNK] == threshold)+lo)[:keep-cursor]
        hot[cursor:cursor+len(ids)] = ids; cursor += len(ids)
        if cursor == keep: break
    fallback = capacity-cursor
    if fallback:
        require(revpr_scores is not None and len(revpr_scores) == capacity,
                'RevPR scores required; never fill by zero-frequency node IDs')
        eligible = np.flatnonzero(counts[revpr_hot] == 0)
        require(len(eligible) >= fallback, 'RevPR fallback does not cover vacancies')
        scores = np.asarray(revpr_scores)[eligible]
        require(np.isfinite(scores).all() and np.all(scores >= 0), 'Invalid RevPR ranks')
        rank_order = np.lexsort((revpr_hot[eligible], -scores))[:fallback]
        hot[cursor:] = revpr_hot[eligible[rank_order]]
    hot.sort()
    require(len(hot) == capacity and np.all(hot[1:] > hot[:-1]), 'Hot set not unique')
    selected_counts = counts[hot]
    return hot, dict(capacity_rows=capacity, observed_nodes=observed,
                     positive_selected=keep, zero_count_selected=fallback,
                     zero_count_fraction=fallback/capacity, fallback='RevPR rank, then logical ID',
                     threshold_count=threshold if keep else None,
                     min_count=int(selected_counts.min()), max_count=int(selected_counts.max()),
                     total_profile_hits=int(selected_counts.sum()), hot_sha256=digest_array(hot))


def audit(canonical, nodes, sampler_factory, hot_sets, event=lambda **kw: None, check=lambda: None):
    require(0 < len(hot_sets) <= 8, 'Audit bitset capacity')
    flags = np.zeros(nodes, np.uint8)
    for bit, hot in enumerate(hot_sets.values()):
        require(np.all(hot[1:] > hot[:-1]) and hot[0] >= 0 and hot[-1] < nodes, 'Invalid audit hot set')
        flags[hot] |= 1 << bit
    results = []
    for epoch, seed in enumerate(AUDIT_SEEDS):
        sample = sampler_factory(seed); total = 0; hits = [0]*len(hot_sets)
        root_hash = hashlib.sha256(); started = time.perf_counter(); roots_seen = 0
        for i, roots in enumerate(batches(canonical, epoch)):
            check(); ids = logical_ids(sample(roots, i), nodes); mask = flags[ids]
            total += len(ids); roots_seen += len(roots); root_hash.update(roots.tobytes())
            for bit in range(len(hits)): hits[bit] += int(np.count_nonzero(mask & (1 << bit)))
            if i % 100 == 0:
                event(stage='cache_eligibility_audit', epoch=epoch, seed=seed, batch=i+1,
                      batches=math.ceil(len(canonical)/BATCH), logical_requests=total)
        require(roots_seen == len(canonical), 'Incomplete audit epoch')
        results.append(dict(epoch=epoch, seed=seed, roots=roots_seen,
                            roots_sha256=root_hash.hexdigest(), logical_requests=total,
                            seconds=time.perf_counter()-started,
                            caches={name:dict(eligible_rows=h, fraction=h/total) for name,h in zip(hot_sets,hits)}))
    return dict(complete=True, epochs=results, same_trace_for_all_hot_sets=True,
                scope='CPU eligibility on identical DiGiT logical requests before GPU filtering; not native serving',
                gpu_cache_allocated=False, feature_reads=0, optimizer_updates=0)
