"""Exact logical CPU rows with replica aliases; physical proxy values cannot alias."""
from candidates.pa_sage_cache_policy_v1.mapping import primary_rows,chunks
from .common import require,digest
from .geometry import ROW_BYTES


def install(backend,arm,hot,primary,storage,feature_mode='logical_synthetic',chunk_rows=1048576):
    require(feature_mode=='logical_synthetic','Physical proxy replicas cannot share a logical CPU slot')
    require(arm['gpu_policy'] in ('legacy','fifo') and arm['gpu_feature_cache_bytes']>0,'UKS pair retains GPU FIFO')
    require(len(hot)==arm['cpu_rows'],'Wrong exact CPU row budget')
    rows=primary_rows(hot,primary,storage,'logical_node_real')
    backend.begin_exact_cpu_cache(rows,len(storage),ROW_BYTES)
    for lo,slots in chunks(storage,hot,len(primary),chunk_rows):backend.write_cpu_row_map(lo,slots)
    backend.finish_exact_cpu_cache();backend.configure_gpu_cache(arm['gpu_policy'],arm['gpu_feature_cache_bytes'])
    return dict(cpu_rows=len(hot),cpu_feature_bytes=len(hot)*ROW_BYTES,row_map_gpu_bytes=len(storage)*4,
        logical_hot_sha256=digest(hot),gpu_feature_cache_bytes=arm['gpu_feature_cache_bytes'],gpu_policy=arm['gpu_policy'],
        all_replicas_mapped=True,padding_uncached=True,feature_semantics=feature_mode)
