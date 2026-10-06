"""Only normal-exit, post-guard-accepted writes can authorize native training."""
import hashlib,json
from pathlib import Path
from . import protocol as P

def accepted_storage():
    runs=sorted(P.OUT.glob('stage_storage_run_*'))
    if not runs:raise RuntimeError('UKL SSD writer has not run')
    path=runs[-1]/'acceptance.json'
    if not path.is_file():raise RuntimeError('UKL SSD writer has no final acceptance')
    a=json.loads(path.read_text())
    if not (a['passed'] and a['kernel_monitor_ok'] and a['post_guard_passed'] and a['io_preflight_passed'] and P.worker_passed(a['worker_state'],a['worker'])):
        raise RuntimeError('UKL SSD write/exit guard not accepted')
    plan,bound,_=P.inputs(False)
    return dict(accepted=True,acceptance_path=str(path),acceptance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_binding_sha256=P.F.digest(bound),device=plan['expected_device'],arms=bound['arms'],sample_readback_passed=True,
                full_readback_passed=False,raw_ssd_bound=True)
