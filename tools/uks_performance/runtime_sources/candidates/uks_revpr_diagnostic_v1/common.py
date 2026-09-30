"""Reuse accepted UKS data/backend with an independent performance protocol."""
from candidates.uks_native_v1.common import *
from candidates.uks_native_v1 import common as native
from candidates.uks_native_retry_v1 import common as repair
HERE=ROOT/'candidates/uks_revpr_diagnostic_v1'
OUT=ROOT/'results/uks_revpr_diagnostic_20260929_v1'
PYTHON='/home/embed/miniconda3/envs/gids/bin/python'

def verify():
    m=read(HERE/'manifest.json');require(repair.verify()==m['repair_sha256'],'Repair source changed')
    for name,value in m['files'].items():require(sha(HERE/name)==value,'Performance source changed: '+name)
    for name,value in m['dependencies'].items():require(sha(ROOT/name)==value,'Dependency changed: '+name)
    return sha(HERE/'manifest.json')

def check_ready():
    verify();repair.check_reuse();binary_receipt()
    s=read(repair.OUT/'summary.json');require(s['passed'] and s['native_short_accepted'],'Native shorts incomplete')
    for stage,h in s['completed'].items():
        d=repair.OUT/stage;a=read(d/'accepted.json')
        require(sha(d/'accepted.json')==h and a['normal_exit'] and read(d/'exit.json')['returncode']==0 and sha(d/'report.json')==a['report_sha256'],'Short evidence changed')
    require(set(s['completed'])=={'smoke_gids','smoke_digit'},'Missing native arm')
    return True

def load_plan(arm):
    # Prior full readback + frozen receipt hashes + current bound source identity.
    # Do not rehash hundreds of GiB in each new performance worker.
    from dataclasses import fields
    from candidates.uks_native_v1.binding import check,protocol
    from candidates.uks_native_v1.storage import api
    from candidates.uks_native_retry_v1.receipts import validate_saved_receipt
    b=check();p=protocol();a=api();folder=native.OUT/('storage_'+arm);v=read(folder/'write_plan.json')
    plan=a.PayloadPlan(**{f.name:v[f.name] for f in fields(a.PayloadPlan)})
    file=DATA/('synthetic' if arm=='gids' else 'payload')/'features.npy'
    require(plan.feature_file==str(file) and plan.feature_file_sha256==b['files'][str(file)]['sha256'] and plan.device_offset_bytes==p['ssd_offsets'][arm] and plan.feature_dim==256,'Wrong reused feature plan')
    validate_saved_receipt(folder/a.VERIFY_RECEIPT,plan)
    return plan
