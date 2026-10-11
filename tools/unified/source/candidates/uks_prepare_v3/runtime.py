"""256D GIDS connection; a future native worker must supply accepted inputs and locks."""
import importlib
import os
import sys
from pathlib import Path
from .common import ROOT,HERE,BINARY,require,heavy_gate,binary_receipt
from .geometry import extent
from .backend import allocation,install_on_loader


def arm_budget(p,arm):
    require(arm in p['arms'],'Unknown arm')
    return dict(p['arms'][arm],cpu_rows=p['cpu_cache_rows'],cpu_feature_bytes=p['cpu_cache_rows']*1024,
        gpu_feature_cache_bytes=p['gpu_cache_bytes'])


def loader_kwargs(p,arm,storage_rows,pool,geometry):
    require(p['features']['mode']=='logical_synthetic' and pool['feature_mode']=='logical_synthetic',
            'Logical CPU aliases require identical logical synthetic features, not physical proxy rows')
    require(pool['passed'] is True and pool['row_bytes']==1024,'Missing accepted UKS 256D feature-pool receipt')
    require(geometry.feature_row_bytes==1024 and geometry.cache_slot_bytes==4096 and
            geometry.minimum_transfer_bytes==4096 and geometry.group_size==2,'Wrong UKS I/O geometry')
    extent(storage_rows,pool['offset'],pool['verified_bytes'])
    a=allocation(arm_budget(p,arm))
    return dict(page_size=4096,off=pool['offset'],cache_dim=256,num_ele=storage_rows*256,
        num_ssd=1,ssd_list=[0],cache_size=a['gpu_dma_allocation_bytes']//2**20,ctrl_idx=0,
        window_buffer=False,accumulator_flag=False,feature_index_mode='explicit',gpu_cache_policy=p['arms'][arm]['gpu_policy'],
        cpu_feature_path='mapped',mixed_io=True,device_io_stats=True,
        mixed_io_geometry=geometry.with_payload_offset(pool['offset']).to_native_mapping())


def setup_sampling_imports():
    # CPU sampler/code use only. Importing DGL/CUDA is the caller's responsibility.
    from ae.pa_sage.common import setup_imports
    # The PA helper selects 16 threads. Preserve the caller's bounded CPU
    # environment before any framework import, including repeat calls.
    names=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
    previous={key:os.environ.get(key) for key in names}
    try:
        setup_imports()
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    sys.path.insert(0,str(ROOT/'candidates/pa_sage_bidir_native_v2/runtime'))


def prepare_imports():
    heavy_gate();binary_receipt()
    require(not any(k=='GIDS' or k.startswith('GIDS.') or k.startswith('BAM_Feature_Store') for k in sys.modules),
            'UKS device work requires a fresh process')
    setup_sampling_imports();sys.path.insert(0,str(HERE/'runtime'))
    module=importlib.import_module('BAM_Feature_Store.BAM_Feature_Store')
    require(Path(module.__file__).resolve()==BINARY.resolve() and module.cache_policy_abi()==4 and
            module.cache_policy_row_bytes()==1024,'Wrong UKS native library')
    return importlib.import_module('GIDS').GIDS,module


def create_loader(p,arm,storage_rows,pool,geometry,hot,primary,storage):
    heavy_gate()
    kwargs=loader_kwargs(p,arm,storage_rows,pool,geometry)
    cls,module=prepare_imports();loader=cls(**kwargs)
    receipt=install_on_loader(loader,module,arm_budget(p,arm),hot,primary,storage)
    return loader,receipt
