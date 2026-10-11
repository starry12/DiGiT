"""Pinned successful UKL v25 runs and CL optimization evidence, then local micro."""
import hashlib,json
from pathlib import Path
from . import protocol as P

def prior_accepted():
    pins=json.loads((P.BASE_OUT/'predecessor.json').read_text())
    for rel,digest in pins.items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts:raise RuntimeError('Invalid predecessor path')
        f=P.ROOT/p
        if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Accepted predecessor changed: '+rel)
    q=json.loads((P.ROOT/'results/ukl_mixed512_20261004_v25/gpu/latest_status.json').read_text())
    if not(q['passed'] and q['runs_complete']==10 and q['rounds_complete']==5):
        raise RuntimeError('UKL accepted five-round predecessor required')
    from .storage import accepted_storage
    storage_sha=accepted_storage()['acceptance_sha256']
    for run in q['runs']:
        path=Path(run['path']);a=json.loads(path.read_text());w=a['worker']
        rel=str(path.relative_to(P.ROOT))
        if rel not in pins or hashlib.sha256(path.read_bytes()).hexdigest()!=run['sha256']:
            raise RuntimeError('Unpinned UKL acceptance')
        if not(a['passed'] and a['kernel_monitor_ok'] and a['post_guard_passed'] and a['gpu_released']
               and a['worker_state']['Result']=='success' and a['worker_state']['ExecMainStatus']=='0'
               and w['updates']==320 and w['storage_acceptance_sha256']==storage_sha):
            raise RuntimeError('UKL predecessor lifecycle or storage mismatch')
    for name in ('cl_sampling_opt_20261005_v14r2','cl_common_gpu_20261005_v16'):
        c=json.loads((P.ROOT/'results'/name/'completion_review.json').read_text())
        if not c['passed']:raise RuntimeError('CL optimization not accepted')
    return hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()

def require_smoke():
    prior=prior_accepted()
    path=P.BASE_OUT/'common_gpu_micro_acceptance.json';r=json.loads(path.read_text())
    if not (r['passed'] and r['normal_exit'] and r['gpu_released'] and r['kernel_ok']
            and r['graph_released'] and r['raw_ssd_access'] is False
            and r['cases']==210 and r['block_cases']==18 and r['training_steps']==18
            and r['buffers_reused'] and r['gids_native_unchanged'] and r['digit_native_optimized']
            and r['manifest_sha256']==P.verify_manifest()):
        raise RuntimeError('Accepted two-arm GPU parity required')
    return hashlib.sha256((prior+hashlib.sha256(path.read_bytes()).hexdigest()).encode()).hexdigest()
