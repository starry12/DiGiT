"""Require the pinned, normally exited CL v11 full-graph smoke pair."""
import hashlib
import json
from pathlib import Path
from . import protocol as P
from candidates.cl_training_v11 import protocol as parent


def prior_smoke():
    binding=P.BASE_OUT/'qualification.json'
    pins=json.loads(binding.read_text())
    for name,digest in pins.items():
        p=Path(name)
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Pinned CL short acceptance changed: '+name)
    micro=parent.require_micro()
    queue_path=parent.BASE_OUT/'queue_status.json'
    q=json.loads(queue_path.read_text())
    if str(queue_path) not in pins or not (q['passed'] and q['gpu_smoke_accepted'] and q['stage']=='complete'):
        raise RuntimeError('Completed CL smoke pair required')
    for arm in ('gids','digit'):
        r=q['smoke']['arms'][arm];p=Path(r['path'])
        if pins.get(str(p))!=r['sha256']:
            raise RuntimeError('Smoke arm is not pinned')
        a=json.loads(p.read_text());w=a['worker']
        if not (a['passed'] and a['stage']==arm and a['pair_id']==q['smoke']['run_id']
                and a['selected_gpu']==q['smoke']['selected_gpu']
                and a['worker_state']['Result']=='success' and a['worker_state']['ExecMainStatus']=='0'
                and a['gpu_released'] and a['kernel_monitor_ok'] and a['post_guard_passed']
                and w['updates']==4 and w['warmup_batches']==0 and w['measured_batches']==4
                and w['graph_and_aux_released'] and w['external_cache_released']
                and w['finite'] and not a['raw_ssd_writes']
                and a['predecessor']['micro_sha256']==micro):
            raise RuntimeError('CL smoke lifecycle evidence incomplete: '+arm)
    return hashlib.sha256(binding.read_bytes()).hexdigest()


def require_smoke():
    prior=prior_smoke()
    path=P.BASE_OUT/'optimized_micro_acceptance.json';r=json.loads(path.read_text())
    if not (r.get('passed') is True and r['normal_exit'] and r['gpu_released'] and r['kernel_ok']
            and r['cases']==210 and r['block_cases']==18 and r['training_steps']==9
            and r['graph_released'] and r['raw_ssd_access'] is False
            and r['manifest_sha256']==P.verify_manifest()):
        raise RuntimeError('Accepted optimized CUDA parity required')
    return hashlib.sha256((prior+hashlib.sha256(path.read_bytes()).hexdigest()).encode()).hexdigest()
