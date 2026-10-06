"""Validate exclusive accounting and produce one accepted diagnostic report."""
import hashlib
import json
import math
from pathlib import Path
from .instrument import KEYS

GROUPS={
    'native_sampling':('native_cuda_entry',),
    'native_buffers_and_transfer':('native_buffers_and_transfer',),
    'cpu_owner_validation':('cpu_owner_validation',),
    'cpu_group_validation':('cpu_group_validation',),
    'cpu_output_validation':('cpu_output_validation',),
    'cpu_frontier_block':('cpu_frontier','cpu_block'),
    'gpu_postprocess':('gpu_numbering','gpu_block','gpu_storage_rows'),
    'sampling_other':('sampling_other','frontier_other','native_sample_other'),
    'request_mapping':('request_mapping',),
    'feature_fetch':('feature_fetch',),
    'block_label_transfer':('block_label_transfer',),
    'forward_loss':('forward_loss',),
    'backward':('backward',),
    'zero_grad_adam':('zero_grad','adam'),
    'runtime_check':('runtime_check',),
    'batch_other':('batch_other',),
}


def valid_profile(p,arm,wall):
    try:
        if arm not in ('gids','digit') or not math.isfinite(wall) or wall<=0:return False
        if not (p['schema']=='cl-symmetric-stages-v1' and p['measured_batches']==300
                and p['synchronized_wall'] is True and p['pure_kernel_timing'] is False):return False
        if any(not math.isfinite(p[k]) or p[k]<0 for k in ('exclusive_sum_seconds','batch_wall_seconds','boundary_sync_seconds')):return False
        s=p['stages']
        if set(s)!=set(KEYS):return False
        for v in s.values():
            if type(v['calls'])is not int or v['calls']<0:return False
            if any(not math.isfinite(v[k]) or v[k]<0 for k in ('inclusive_seconds','exclusive_seconds','thread_cpu_exclusive_seconds','boundary_sync_seconds')):return False
            if v['exclusive_seconds']>v['inclusive_seconds']+1e-6:return False
        expected={'batch_other':300,'sampling_other':300,'frontier_other':300,
                  'native_sample_other':900,'native_cuda_entry':900,'native_buffers_and_transfer':900,
                  'cpu_owner_validation':900,'cpu_output_validation':900,'request_mapping':300,
                  'feature_fetch':300,'block_label_transfer':300,'forward_loss':300,
                  'zero_grad':300,'backward':300,'adam':300,'runtime_check':600,
                  'gpu_numbering':900 if arm=='digit' else 0,
                  'gpu_block':900 if arm=='digit' else 0,
                  'gpu_storage_rows':300 if arm=='digit' else 0,
                  'cpu_frontier':900 if arm=='gids' else 0,
                  'cpu_block':900 if arm=='gids' else 0}
        if any(s[k]['calls']!=v for k,v in expected.items()):return False
        if arm=='gids' and s['cpu_group_validation']['calls']!=0:return False
        if arm=='digit' and s['cpu_group_validation']['calls']<300:return False
        total=sum(v['exclusive_seconds'] for v in s.values())
        if abs(total-p['exclusive_sum_seconds'])>1e-6:return False
        if abs(total-p['batch_wall_seconds'])>max(1e-6,total*1e-9):return False
        if abs(total-s['batch_other']['inclusive_seconds'])>1e-6:return False
        return total>0 and total<=wall+1e-6
    except (KeyError,TypeError,ValueError):return False


def review(series):
    if not (series['passed'] and series['runs_complete']==2 and series['rounds_complete']==1):
        raise ValueError('One complete accepted diagnostic pair required')
    rows={}
    for arm in ('gids','digit'):
        ref=series['pairs'][0][arm];path=Path(ref['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise RuntimeError('Receipt changed')
        a=json.loads(path.read_text());w=a['worker'];p=w['stage_profile']
        if not (a['passed'] and a['gpu_released'] and a['post_guard_passed'] and a['kernel_monitor_ok']
                and a['pair_id']==series['pairs'][0]['pair_id'] and a['selected_gpu']==series['selected_gpu']
                and valid_profile(p,arm,w['seconds'])):raise RuntimeError('Incomplete profile acceptance')
        rows[arm]=dict(seconds=w['seconds'],profile=p,
                       exclusive_seconds={g:sum(p['stages'][k]['exclusive_seconds'] for k in keys) for g,keys in GROUPS.items()},
                       outside_batch_seconds=max(0.,w['seconds']-p['batch_wall_seconds']),receipt=str(path))
    return dict(passed=True,diagnostic_only=True,performance_claim=False,arms=rows,
                differences_digit_minus_gids={g:rows['digit']['exclusive_seconds'][g]-rows['gids']['exclusive_seconds'][g] for g in GROUPS},
                timing='Synchronized wall, exclusive nested intervals; native entry includes kernel launch/execution/synchronization; not pure kernel timing')
