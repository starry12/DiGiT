"""Explicit AE execution policy: change only the host admission reserve.

Training remains the sealed candidate plus this separately hashed deployment
policy. The candidate's default 320 GiB plan is verified before applying 192 GiB;
GPU budgets, CPU/GPU cache capacities, sampling and training are untouched.
"""
import copy, hashlib, importlib, json
from pathlib import Path
GIB=2**30
OLD_HOST=320*GIB
HOST_REQUIRED=192*GIB
VERSIONS=('cache_v3','graph_v4')

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def policy():
    return dict(id='ae-host-admission-192gib-v1',host_required_bytes=HOST_REQUIRED,
        candidate_default_host_required_bytes=OLD_HOST,source_sha256=sha(__file__),
        worker_launcher_sha256=sha(Path(__file__).with_name('ablation_worker.py')),
        scope='Host startup admission only; no training or cache configuration change')

def apply(plan):
    require(plan['host_required_bytes']==OLD_HOST,'Candidate default host budget changed; audit before applying policy')
    result=copy.deepcopy(plan)
    result['host_required_bytes']=HOST_REQUIRED
    result['host_budget_note']='AE deployment policy: 192 GiB available host RAM at worker admission; historical max RSS ~137.2 GiB; engineering reserve, not a worst-case bound'
    result['host_admission_policy']=policy()
    return result

def install(version):
    require(version in VERSIONS,'Unsupported admission version')
    module=importlib.import_module('candidates.pa_sage_ablation_'+version+'.admission')
    original=module.estimate
    require(not getattr(original,'_host192_policy',False),'Policy already installed')
    def estimate(): return apply(original())
    estimate._host192_policy=True
    module.estimate=estimate
    return original

def validate_receipt(receipt):
    require(receipt['host_required_bytes']==HOST_REQUIRED and receipt['host_admission_policy']==policy(),'Wrong effective host admission policy')
    require(receipt['passed'] and receipt['host_available_bytes']>=HOST_REQUIRED,'Host admission not satisfied')
    require(receipt['free_bytes']>=receipt['required_bytes'],'GPU admission not satisfied')

def audit_estimates():
    result={}
    for version in VERSIONS:
        module=importlib.import_module('candidates.pa_sage_ablation_'+version+'.admission')
        old=module.estimate();new=apply(old)
        changed={k for k in old if old[k]!=new[k]}
        require(changed=={'host_required_bytes','host_budget_note'},'Unexpected candidate plan change')
        require(set(new)-set(old)=={'host_admission_policy'},'Unexpected admission fields')
        module.cfg() # Immutable cache/training protocol remains valid.
        result[version]=dict(old_host_required_bytes=old['host_required_bytes'],host_required_bytes=new['host_required_bytes'],
            gpu_required_bytes=new['required_bytes'],gpu_components_unchanged=new['components_bytes']==old['components_bytes'],
            candidate_sha256=module.verify())
    return dict(passed=True,policy=policy(),plans=result)
