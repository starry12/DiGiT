import importlib
import sys
from .common import *

def prepare_imports():
    heavy_gate();binary_receipt();setup()
    require(not any(k=='GIDS' or k.startswith('BAM_Feature_Store') for k in sys.modules),'Fresh native worker required')
    sys.path.insert(0,str(HERE/'runtime'))
    native=importlib.import_module('BAM_Feature_Store.BAM_Feature_Store')
    require(Path(native.__file__).resolve()==BINARY.resolve() and native.cache_policy_abi()==5 and
            native.cache_policy_row_bytes()==native.cache_policy_page_bytes()==512,'Wrong PA native binary')
    return importlib.import_module('GIDS').GIDS,native

def arm_budget(p,arm):
    return dict(p['arms'][arm],cpu_rows=p['cpu_cache_rows'],cpu_feature_bytes=p['cpu_cache_rows']*512,
                gpu_feature_cache_bytes=p['gpu_cache_bytes'])

def loader_kwargs(p,arm,rows,offset):
    from digit.io_geometry import IOGeometry
    geometry=IOGeometry.create(feature_row_bytes=512,group_size=1,minimum_transfer_bytes=512,target_request_bytes=512)
    return dict(page_size=512,off=offset,cache_dim=128,num_ele=rows*128,num_ssd=1,ssd_list=[0],cache_size=4096,
                ctrl_idx=0,window_buffer=False,accumulator_flag=False,feature_index_mode='explicit',
                gpu_cache_policy=p['arms'][arm]['gpu_policy'],cpu_feature_path='mapped',mixed_io=True,device_io_stats=True,
                mixed_io_geometry=geometry.with_payload_offset(offset).to_native_mapping())
