"""Bounded independent checks of sampled edges, addresses, and DGL blocks."""
import hashlib
import numpy as np

RAW_NAMES = ('src', 'dst', 'counts', 'errors', 'eid', 'rows')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(*arrays):
    result = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        result.update(value.dtype.str.encode())
        result.update(str(value.shape).encode())
        result.update(memoryview(value).cast('B'))
    return result.hexdigest()


def raw_digest(raw):
    return digest(*(raw[key] for key in RAW_NAMES))


def equal_arrays(left, right, label):
    require(set(left) == set(right), label + ' keys differ')
    for key in left:
        require(left[key].dtype == right[key].dtype
                and np.array_equal(left[key], right[key]), label + ' differs: ' + key)


def stable_unique(values):
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


def address_checks(graph, src, dst, rows, grouped, packed):
    """Read only selected primary/member entries and bounded owner-group chunks."""
    arrays = graph.arrays
    require(not len(rows) or (rows.min() >= 0 and rows.max() < graph.storage_rows),
            'Storage address outside declared layout')
    if not packed:
        require(np.array_equal(rows, src.astype(np.int64)), 'GIDS addresses must be logical IDs')
        return
    primary = arrays['primary'][src]
    mismatch = rows != primary
    if not grouped:
        require(not np.any(mismatch), 'Ungrouped packed address differs from primary')
        return
    # Frozen UKL packed512-g2 bases are increasing. Search touches O(log G)
    # entries, then verify every returned base, member, and owner membership.
    # Synthetic fixtures use the same increasing, disjoint base convention.
    selected = np.flatnonzero(mismatch)
    if not len(selected):
        return
    groups = np.searchsorted(arrays['bases'], rows[selected], side='right') - 1
    require(np.all(groups >= 0) and np.all(groups < graph.groups), 'Unmapped group address')
    bases = arrays['bases'][groups]
    slots = rows[selected] - bases
    require(np.all((slots == 0) | (slots == 1)), 'Group address is not a member slot')
    require(np.array_equal(arrays['members'][groups, slots], src[selected]),
            'Group address maps to a different logical node')
    pairs = np.unique(np.column_stack((dst[selected], groups)), axis=0)
    for owner, group in pairs:
        lo, hi = map(int, arrays['gptr'][int(owner):int(owner) + 2])
        require(0 <= lo <= hi <= graph.units, 'Group ownership range invalid')
        found = False
        for start in range(lo, hi, 4096):
            if np.any(arrays['gidx'][start:min(start + 4096, hi)] == group):
                found = True
                break
        require(found, 'Selected group address belongs to another owner')


def raw_checks(graph, seeds, fanout, raw, grouped, packed):
    require(set(raw) == set(RAW_NAMES), 'Native raw output key set changed')
    count = len(seeds)
    for name in RAW_NAMES:
        value = raw[name]
        dtype = np.dtype('int64' if name in ('eid', 'rows') else 'int32')
        require(isinstance(value, np.ndarray) and value.dtype == dtype
                and value.shape == (count if name in ('counts', 'errors') else count * fanout,),
                'Native raw output shape or dtype invalid: ' + name)
    require(not np.any(raw['errors']), 'Native output contains errors')
    counts = raw['counts']
    require(np.all(counts >= 0) and np.all(counts <= fanout), 'Native count exceeds fanout')
    arrays = graph.arrays
    begin, end = arrays['ptr'][seeds], arrays['ptr'][seeds + 1]
    require(np.all(begin >= graph.origin) and np.all(end >= begin)
            and np.all(end <= graph.origin + graph.edges), 'Selected CSC ranges invalid')
    if not grouped:
        require(np.array_equal(counts, np.minimum(end - begin, fanout)),
                'Ungrouped count differs from degree/fanout')
    else:
        require(np.all(counts <= end - begin), 'Grouped output exceeds owner degree')
    mask = np.arange(fanout)[None, :] < counts[:, None]
    compact = {}
    for key in ('src', 'dst', 'eid', 'rows'):
        shaped = raw[key].reshape(count, fanout)
        require(np.all(shaped[~mask] == -1), 'Unused native output is not sentinel: ' + key)
        compact[key] = shaped[mask]
    src, dst, eid, rows = (compact[key] for key in ('src', 'dst', 'eid', 'rows'))
    require(not len(src) or (src.min() >= 0 and src.max() < graph.nodes), 'Source node outside graph')
    require(np.array_equal(dst, np.repeat(seeds, counts)), 'Output destination differs from owner')
    require(np.all(eid >= np.repeat(begin, counts)) and np.all(eid < np.repeat(end, counts)),
            'EID lies outside its owner adjacency')
    require(np.array_equal(arrays['idx'][eid - graph.origin], src), 'EID maps to wrong source node')
    # A native draw samples occurrences without replacement. Sorting <=32
    # values per destination avoids graph-sized uniqueness state.
    sorted_eids = np.sort(np.where(mask, raw['eid'].reshape(count, fanout), -1), axis=1)
    require(not np.any((sorted_eids[:, 1:] == sorted_eids[:, :-1])
                       & (sorted_eids[:, 1:] >= 0)), 'Repeated EID within one destination draw')
    address_checks(graph, src, dst, rows, grouped, packed)
    return compact


def block_arrays(block, dgl):
    u, v = block.edges(order='eid')
    return dict(src=block.srcdata[dgl.NID].numpy(), dst=block.dstdata[dgl.NID].numpy(),
                u=u.numpy(), v=v.numpy(), eid=block.edata[dgl.EID].numpy())


def block_checks(graph, layers, result, roots, packed):
    import dgl
    from candidates.ukl_sage_compact_v1.sampler import STORAGE_ROW
    nodes, targets, blocks = result
    require(len(blocks) == len(layers) == 3, 'Three blocks required')
    require(np.array_equal(nodes.numpy(), layers[0][1])
            and np.array_equal(targets.numpy(), roots), 'Block input/output node identity differs')
    receipts = []
    for index, ((dst, src, compact), block) in enumerate(zip(layers, blocks)):
        expected_src = stable_unique(np.concatenate((dst, compact['src'])))
        require(np.array_equal(src, expected_src), 'Frontier is not stable unique destinations plus neighbors')
        value = block_arrays(block, dgl)
        for key, limit in (('u', len(src)), ('v', len(dst))):
            require(value[key].dtype == np.int64 and value[key].shape == compact['eid'].shape
                    and (not len(value[key]) or (value[key].min() >= 0 and value[key].max() < limit)),
                    'Block local index outside frontier: ' + key)
        require(np.array_equal(value['src'], src) and np.array_equal(value['dst'], dst),
                'Block frontier identity differs')
        require(np.array_equal(value['eid'], compact['eid']), 'Block EID ordering differs')
        require(np.array_equal(value['src'][value['u']], compact['src'])
                and np.array_equal(value['dst'][value['v']], compact['dst']),
                'Block local edge endpoints differ')
        storage = np.empty(0, dtype=np.int64)
        if index == 0:
            storage = block.srcdata[STORAGE_ROW].numpy()
            expected = np.asarray(graph.arrays['primary'][src] if packed else src,
                                  dtype=np.int64).copy()
            lookup = {int(node): at for at, node in enumerate(src)}
            _, first = np.unique(compact['src'], return_index=True)
            for at in first:
                expected[lookup[int(compact['src'][at])]] = compact['rows'][at]
            require(np.array_equal(storage, expected), 'Block storage rows violate first occurrence mapping')
            require(not len(storage) or (storage.min() >= 0 and storage.max() < graph.storage_rows),
                    'Block storage row outside layout')
        receipts.append(dict(block_sha256=digest(*(value[key] for key in ('src', 'dst', 'u', 'v', 'eid'))),
                             storage_sha256=digest(storage)))
    return receipts
