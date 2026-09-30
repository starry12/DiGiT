"""Exact-row installation contract for a future isolated native cache backend.

Existing production binaries are deliberately not patched or selected here.
CPU tests exercise this adapter with a recording backend, not a real NVMe device.
"""
from .common import require, digest
from .mapping import primary_rows, chunks


def required_capabilities(arm):
    return dict(exact_logical_cpu_rows=True, replica_aliases=True, padding_uncached=True,
                useful_io_counters=True, exclusive_serving_counters=True,
                static_gpu_bypass=arm['gpu_policy'] == 'bypass',
                fifo_gpu_first=arm['gpu_policy'] == 'fifo')


def check_backend(capabilities, arm):
    require(arm['gpu_policy'] in ('bypass', 'fifo'), 'Legacy cache is not disabled caching')
    for key, enabled in required_capabilities(arm).items():
        if enabled:
            require(capabilities.get(key) is True, 'Native capability not yet accepted: ' + key)
    require(arm['gpu_feature_cache_bytes'] > 0 if arm['gpu_policy'] == 'fifo'
            else arm['gpu_feature_cache_bytes'] == 0, 'Invalid native cache capacity')


def install(backend, arm, hot_nodes, node_to_primary_row, storage_to_node,
            feature_mode='logical_node_real', chunk_rows=1048576):
    """The backend owns device allocations, pinned CPU memory and preload I/O."""
    check_backend(backend.capabilities(), arm)
    require(len(hot_nodes) == arm['cpu_rows'], 'Selected hot set exceeds/differs from the declared budget')
    rows = primary_rows(hot_nodes, node_to_primary_row, storage_to_node, feature_mode)
    # Pass the exact slot data; no eight-row page expansion of selected nodes.
    backend.begin_exact_cpu_cache(rows, len(storage_to_node), row_bytes=512)
    for lo, slots in chunks(storage_to_node, hot_nodes, len(node_to_primary_row), chunk_rows):
        backend.write_cpu_row_map(lo, slots)
    backend.finish_exact_cpu_cache()
    backend.configure_gpu_cache(arm['gpu_policy'], arm['gpu_feature_cache_bytes'])
    return dict(cpu_rows=len(hot_nodes), cpu_feature_bytes=len(hot_nodes) * 512,
                row_map_gpu_bytes=len(storage_to_node) * 4, logical_hot_sha256=digest(hot_nodes),
                gpu_feature_cache_bytes=arm['gpu_feature_cache_bytes'], gpu_policy=arm['gpu_policy'],
                all_replicas_mapped=True, padding_uncached=True, feature_semantics=feature_mode)
