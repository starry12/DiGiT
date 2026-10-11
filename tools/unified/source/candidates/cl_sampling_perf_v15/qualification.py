"""Reuse the accepted optimized sampler and full-graph pair without rerunning micro."""
import hashlib,json
from pathlib import Path
from . import protocol as P
from candidates.cl_sampling_opt_v14r2.qualification import require_smoke as prior_micro

def require_smoke():
    pins=json.loads((P.BASE_OUT/'predecessor.json').read_text())
    for rel,digest in pins.items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts:raise RuntimeError('Invalid predecessor path')
        f=P.ROOT/p
        if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Accepted predecessor changed: '+rel)
    previous=P.ROOT/'results/cl_sampling_opt_20261005_v14r2'
    micro_sha=prior_micro()
    q=json.loads((previous/'queue_status.json').read_text())
    if not (q['passed'] and q['stage']=='complete' and q['pair']['runs_complete']==2
            and q['pair']['rounds_complete']==1):raise RuntimeError('Accepted optimized pair required')
    for run in q['pair']['runs']:
        p=Path(run['path']);a=json.loads(p.read_text())
        if str(p.relative_to(P.ROOT)) not in pins or hashlib.sha256(p.read_bytes()).hexdigest()!=run['sha256']:
            raise RuntimeError('Unpinned optimized arm')
        if not (a['passed'] and a['kernel_monitor_ok'] and a['post_guard_passed'] and a['gpu_released']
                and a['worker_state']['Result']=='success' and a['worker_state']['ExecMainStatus']=='0'
                and a['worker']['updates']==320):raise RuntimeError('Optimized arm lifecycle failed')
    return hashlib.sha256((micro_sha+json.dumps(pins,sort_keys=True)).encode()).hexdigest()
