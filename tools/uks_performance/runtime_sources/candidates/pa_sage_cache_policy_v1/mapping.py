"""Exact CPU membership: replicas alias one logical slot, padding never hits."""
import numpy as np
from .common import require


def logical_slots(hot, nodes):
    hot = np.asarray(hot)
    require(hot.ndim == 1 and hot.dtype == np.int64 and len(hot) < 2**31, 'Invalid hot IDs')
    require(not len(hot) or (hot[0] >= 0 and hot[-1] < nodes and np.all(np.diff(hot) > 0)), 'Hot IDs must be sorted and unique')
    slots = np.zeros(nodes, dtype=np.uint32)
    slots[hot] = np.arange(1, len(hot) + 1, dtype=np.uint32)
    return slots


def storage_slots(storage_to_node, slots):
    storage = np.asarray(storage_to_node)
    require(storage.ndim == 1 and storage.dtype == np.int64 and
            np.all((storage >= -1) & (storage < len(slots))), 'Invalid storage mapping')
    result = np.zeros(len(storage), dtype=np.uint32)
    valid = storage >= 0
    result[valid] = slots[storage[valid]]
    return result


def chunks(storage_to_node, hot, nodes, chunk_rows=1048576):
    require(type(chunk_rows) is int and chunk_rows > 0, 'Invalid chunk size')
    slots = logical_slots(hot, nodes)
    for lo in range(0, len(storage_to_node), chunk_rows):
        yield lo, storage_slots(storage_to_node[lo:lo + chunk_rows], slots)


def primary_rows(hot, node_to_primary_row, storage_to_node, feature_mode):
    require(feature_mode == 'logical_node_real', 'Cannot alias arbitrary physical-row proxy features')
    hot = np.asarray(hot)
    logical_slots(hot, len(node_to_primary_row))
    rows = np.asarray(node_to_primary_row[hot], dtype=np.int64)
    require(not len(rows) or (rows.min() >= 0 and rows.max() < len(storage_to_node)), 'Primary address out of range')
    require(np.array_equal(storage_to_node[rows], hot), 'Primary address inverse differs')
    return rows
