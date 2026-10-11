"""Common reusable GPU buffers/postprocessing; each arm keeps its native sampler."""
from candidates.ukl_native_sampling_v10.sampling import Graph
from candidates.ukl_native_sampling_v10 import sampling as frozen
from .protocol import VARIANT

def select(arm):
    if arm == 'gids':
        from candidates.ukl_postprocess_v22.device import Native,Sampler
        return Native,Sampler
    if arm != 'digit':
        raise ValueError('Unknown arm')
    if VARIANT == 'host':
        from candidates.ukl_postprocess_v22.host import Sampler
        return frozen.Native, Sampler
    if VARIANT == 'buffer':
        from candidates.ukl_postprocess_v22.buffer import Native
        from candidates.ukl_postprocess_v22.host import Sampler
        return Native, Sampler
    from .device import Native, Sampler
    return Native, Sampler

def effective_variant(arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    return 'gids_gpu_common_v26' if arm == 'gids' else VARIANT
