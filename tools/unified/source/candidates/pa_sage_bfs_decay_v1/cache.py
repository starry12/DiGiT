"""Identical original DiGiT hot set and preload in all three ordering arms."""
from .common import require, sha


def validate_binding(value, arm, p, binding):
    require(value == dict(rows=p['cpu_cache_rows'],
        cpu_feature_bytes=p['cpu_cache_rows']*512,
        hot_sha256=binding['files'][p['base_layout']+'/full_cpu_rows.npy']['sha256'],
        mapping='page', row_map_gpu_bytes=0, gpu_feature_bytes=p['gpu_cache_bytes'],
        padding_uncached=True), 'DiGiT feature cache changed')


def install_cache(loader, arm, base, bundle, p, torch, np, binding, storage_rows):
    path=base/'full_cpu_rows.npy'
    hot=np.load(path,mmap_mode='r')
    require(len(hot)==p['cpu_cache_rows'], 'CPU budget changed')
    loader.cpu_backing_buffer(128,len(hot))
    loader.set_cpu_buffer(torch.from_numpy(hot.copy()),len(hot))
    result=dict(rows=len(hot),cpu_feature_bytes=len(hot)*512,hot_sha256=sha(path),
                mapping='page',row_map_gpu_bytes=0,gpu_feature_bytes=p['gpu_cache_bytes'],
                padding_uncached=True)
    validate_binding(result,arm,p,binding)
    return result,dict(primary_rows=hot)
