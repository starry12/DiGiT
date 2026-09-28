import math
from candidates.ig_sage_stage_profile_v1.common import *
def estimate():
    p=cfg();rows=read(DATA/'full/manifest.json')['feature']['num_storage_rows']
    parts=dict(feature_store_range=rows*32,gpu_cache=p['gpu_cache_bytes'],cache_metadata=512*2**20,
               useful_bitmaps=p['gpu_cache_bytes']//4096*8+48,torch_model_blocks_features_allowance=p['model_probe_limit_bytes'],runtime_reserve=4*2**30)
    return dict(schema='digit-ig-perf-admission-v1',required_bytes=math.ceil(sum(parts.values())*1.2),components_bytes=parts,
                host_required_bytes=p['host_required_bytes'],original_csc_gpu_bytes=0,
                original_csc_host_bytes=8*(p['nodes']+1+2*p['edges']),host_metadata_total_bytes=86753666496,
                cpu_cache_bytes=p['cpu_cache_rows']*4096,formal_bound=False,reservation=False,
                note='Engineering envelope for both systems and all three models, verified with maximum-block GPU probe and native smoke. Original CSC shared on pinned host; 320GiB includes ~103GiB CPU cache, ~81GiB CSC+metadata, mmap/chunks/order/labels and reserve.')
def check_live():
    import torch
    p=estimate();free,total=torch.cuda.mem_get_info();p.update(free_bytes=free,total_bytes=total,host_available_bytes=host())
    p['passed']=free>=p['required_bytes'] and host()>=p['host_required_bytes'];return p
