"""CPU-only check in the real service namespace; never opens GPU or SSD."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_context
model_context.activate()
import sys
sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
import stable_identity
_identity_registry=stable_identity.install('UKS')
import sys,os,importlib
from pathlib import Path
ROOT=Path('/home/embed/digit');sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,str(ROOT))
from candidates.uks_freq_bfs_retry_v1.common import check_ready,load_roots,load_plan
check_ready()
from candidates.uks_native_v1.binding import check,protocol
check();protocol()
from candidates.uks_native_v1.storage import api,state_path
for arm in ('gids','digit'):
    assert len(load_roots(arm))==320*1024
    plan=load_plan(arm);api()._validate_active_state(state_path(plan),plan,'verified')
from candidates.uks_mixed_1k2k_v1.runtime import prepare_imports
prepare_imports()
from candidates.uks_group_incremental_v1.common import raw_extension
raw_extension()
for name in ('candidates.uks_freq_bfs_retry_v1.worker','candidates.uks_freq_bfs_retry_v1.smoke',
             'candidates.uks_freq_bfs_retry_v1.controller','ae.igb.models','candidates.ig_sage_affinity_pair_v1.affinity'):
    importlib.import_module(name)
from candidates.ig_sage_affinity_pair_v1.affinity import apply_affinity
apply_affinity('gids')
from candidates.uks_native_v1.model import create
p=protocol();p.update(fixture=True,nodes=16);create(p,device='cpu')
import torch
assert not torch.cuda.is_initialized()
print('PASS: UKS input identities, SSD receipts, native imports; CUDA not initialized')

import gpu_select,worker,controller
print("PASS: automatic-GPU service adapters import without device allocation")

model_context.selftest()
