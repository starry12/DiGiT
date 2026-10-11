"""Same bounded compact graph ABI for both arms; grouping remains DiGiT-only."""
from candidates.ukl_compact_cover_v1.owned_sampling import Graph

def select(arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    from .device import Native,Sampler
    return Native,Sampler

def effective_variant(arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    return 'gids_gpu_common_v26' if arm=='gids' else 'gpu'
