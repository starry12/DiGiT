"""Reuse the accepted optimized sampler and full-graph pair without rerunning micro."""
import hashlib,json
from pathlib import Path
from . import protocol as P
from candidates.cl_sampling_opt_v14r2.qualification import require_smoke as prior_micro

def prior_accepted():
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


def require_smoke():
    prior=prior_accepted()
    path=P.BASE_OUT/'common_gpu_micro_acceptance.json';r=json.loads(path.read_text())
    if not (r['passed'] and r['normal_exit'] and r['gpu_released'] and r['kernel_ok']
            and r['graph_released'] and r['raw_ssd_access'] is False
            and r['cases']==105 and r['block_cases']==9 and r['training_steps']==9
            and r['buffers_reused'] and r['original_native_sampler']
            and r['manifest_sha256']==P.verify_manifest()):
        raise RuntimeError('Accepted common GPU parity required')
    return hashlib.sha256((prior+hashlib.sha256(path.read_bytes()).hexdigest()).encode()).hexdigest()
