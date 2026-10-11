"""Arm-specific selection: frozen baseline GIDS, optimized DiGiT only."""
from candidates.ukl_native_sampling_v10.sampling import Graph
from candidates.ukl_native_sampling_v10 import sampling as frozen
from .protocol import VARIANT

def select(arm):
    if arm == 'gids':
        return frozen.Native, frozen.Sampler
    if arm != 'digit':
        raise ValueError('Unknown arm')
    if VARIANT == 'host':
        from candidates.ukl_postprocess_v22.host import Sampler
        return frozen.Native, Sampler
    if VARIANT == 'buffer':
        from candidates.ukl_postprocess_v22.buffer import Native
        from candidates.ukl_postprocess_v22.host import Sampler
        return Native, Sampler
    from candidates.ukl_postprocess_v22.device import Native, Sampler
    return Native, Sampler

def effective_variant(arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    return 'baseline_v21' if arm == 'gids' else VARIANT
