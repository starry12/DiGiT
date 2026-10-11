"""Fresh resources using the accepted UKS backend, layout and full-epoch hot sets."""
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from candidates.uks_freq_bfs_retry_v1.common import *
from candidates.uks_native_v1.binding import protocol,check
from candidates.uks_group_first_cpu2_v1.sampling import graph_sampler
from candidates.uks_pa_fast_v1.common import inputs

def create(arm,folder,canonical,source,binding,lock_fds):
    from .guard import RuntimeGuard
    guard=RuntimeGuard(lock_fds)
    variant='gids_default' if arm=='gids' else 'digit_cpu2';optimized=arm=='digit'
    hot_path=lambda a:Path(binding['hot_sets'][a]['path'])
    from candidates.uks_pa_fast_v1.runtime import prepare_imports,loader_kwargs,arm_budget
    cls,module=prepare_imports()
    import torch,dgl
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP
    from digit.io_geometry import IOGeometry
    from candidates.uks_pa_fast_v1.backend import install_on_loader
    from candidates.uks_mixed_1k2k_v1.counters import snapshot,interval
    from candidates.uks_native_v1.model import create
    from candidates.uks_freq_bfs_retry_v1.common import load_plan,check_ready
    from candidates.ig_sage_affinity_pair_v1.affinity import apply_affinity,snapshot as affinity_snapshot,validate_affinity
    affinity=apply_affinity('digit_full' if variant=='digit_cpu2' else 'gids')
    began_setup=time.perf_counter()
    p=__import__('copy').deepcopy(protocol());p['optimizer']['weight_decay']=0.0;b=check()
    from .common import config
    expected=config('short1')
    require(all(p[k]==expected[k] for k in ('seed','fanouts','hidden','layers','dropout')),'UKS model configuration drift')
    require(p['classes']==expected['data']['classes'] and p['gpu_cache_bytes']==expected['gpu_cache_bytes'],'UKS class/cache configuration drift')
    require(p['optimizer']['lr']==expected['lr'] and p['optimizer']['weight_decay']==expected['data']['weight_decay'],'UKS optimizer configuration drift')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);dgl.seed(0);dgl.random.seed(0)
    graph,arrays,artifact,sampler=graph_sampler(arm,0,'gids' if arm=='gids' else 'first')
    if optimized:
        from candidates.uks_pa_fast_v1.sampling import optimize
        sampler=optimize(sampler)
    geometry=IOGeometry.create(feature_row_bytes=1024,group_size=1,minimum_transfer_bytes=1024,target_request_bytes=1024)
    rows=p['nodes'] if arm=='gids' else b['receipts']['g2']['num_storage_rows']
    from candidates.uks_native_v1.storage import plan,api,state_path
    v=load_plan(arm);api()._validate_active_state(state_path(v),v,'verified')
    from candidates.uks_native_retry_v1.receipts import validate_saved_receipt
    verified=validate_saved_receipt(native_parent.OUT/('storage_'+arm)/api().VERIFY_RECEIPT,v)
    pool=dict(passed=True,feature_mode='logical_synthetic',row_bytes=1024,offset=v.device_offset_bytes,verified_bytes=v.payload_bytes)
    loader=cls(**loader_kwargs(p,arm,rows,pool,geometry))
    if arm=='gids':
        hot=np.load(hot_path(arm),mmap_mode='r');primary=np.arange(p['nodes'],dtype=np.int64);storage=primary
    else:
        hot=np.load(hot_path(arm),mmap_mode='r');primary=artifact.arrays['node_to_primary_row'];storage=artifact.arrays['storage_to_node']
    installed=install_on_loader(loader,module,arm_budget(p,arm),hot,primary,storage)
    net,opt,initial=create(p,device='cuda');labels=np.load(DATA/'synthetic/labels.npy',mmap_mode='r')
    from candidates.unified_training_v2.order import roots_digest
    from candidates.uks_mixed_1k2k_v1.counters import snapshot
    before=snapshot(loader.BAM_FS)
    require(before['gpu']['resident_pages']==0,'GPU cache must start cold')
    receipt=dict(source_sha256=source,graph_sha256=__import__('hashlib').sha256(__import__('json').dumps(b,sort_keys=True).encode()).hexdigest(),
        split_sha256=binding['training']['sha256'],canonical_sha256=roots_digest([canonical]),
        cpu_feature_bytes=installed['cpu_feature_bytes'],gpu_feature_bytes=installed['gpu_feature_cache_bytes'],
        worker_id=str(folder),evidence_scope='native',roots_verified_exact=True,gpu_resident_rows_start=0)
    return SimpleNamespace(model=net,optimizer=opt,loader=loader,sampler=sampler,graph=graph,
        labels=labels,features=np.load(DATA/'synthetic/features.npy',mmap_mode='r'),
        owners=(arrays,artifact,hot,primary,storage),receipt=receipt,output=folder,
        snapshot=lambda:snapshot(loader.BAM_FS),interval=interval,check=guard.check,cache=installed)
