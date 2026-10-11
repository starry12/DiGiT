"""One-time production dispatch; no per-iteration wrapper or profiler pool."""
from ig_common import D,binding,require
from pathlib import Path

def expected_configuration(arm):
    require(arm in ('gids','full'),'Unsupported arm')
    module=next(D.glob('IGGroupWarpCUDA*.so'))
    return dict(arm=arm,module=binding(module),implementation='warp_cooperative' if arm=='full' else 'standard_dgl',
                optimized=arm=='full',cooperative=arm=='full',bucket_enabled=False,profile_pool_initialized=False)

def configure(arm):
    from uva_sampler import native
    expected=expected_configuration(arm)
    require(Path(native.__file__).resolve()==Path(expected['module']['path']).resolve() and native.WARP_FALLBACK_API==1,'Wrong sampler extension')
    require(not native.profile_pending(),'Sampler already armed')
    native.set_optimized(arm=='full');native.set_cooperative(arm=='full');native.set_bucket(False,-1,2**63-1)
    require(native.get_optimized()==(arm=='full') and not native.profile_pending(),'Wrong sampler state')
    return expected
