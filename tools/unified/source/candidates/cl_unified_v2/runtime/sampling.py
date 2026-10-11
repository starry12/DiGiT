from candidates.ukl_native_sampling_v10.sampling import Graph

def select(arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    from .device import Native,Sampler
    return Native,Sampler

def effective_variant(arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    return 'gids_gpu_common_v16' if arm=='gids' else 'gpu'
