"""Compare only accepted pair totals; never reconstruct nonexistent stage timings."""
import hashlib,json
from pathlib import Path
from .timing import valid_timing

def review(series):
    if not (series['passed'] and series['rounds_complete']==1 and series['runs_complete']==2
            and series['profiling_run'] is False and series['performance_run'] is True):
        raise RuntimeError('Complete unprofiled pair required')
    current={}
    for run in series['runs']:
        path=Path(run['path']);a=json.loads(path.read_text());w=a['worker']
        if not (hashlib.sha256(path.read_bytes()).hexdigest()==run['sha256'] and a['passed']
                and a['pair_id']==run['pair_id'] and a['pair_id']==series['run_id']+'_r1'
                and a['gpu_released'] and a['kernel_monitor_ok'] and a['post_guard_passed']
                and a['worker_state']['Result']=='success' and a['worker_state']['ExecMainStatus']=='0'
                and valid_timing(w) and w['updates']==320):raise RuntimeError('Invalid unprofiled arm receipt')
        current[run['arm']]=w['seconds']
    if set(current)!={'gids','digit'}:raise RuntimeError('Both arms required')
    from . import protocol as P
    old=json.loads((P.ROOT/'results/ukl_mixed512_20261004_v25/gpu/latest_status.json').read_text())
    best=next(p for p in old['pairs'] if p['round']==old['best_round'])
    before={arm:best[arm]['seconds'] for arm in ('gids','digit')}
    return dict(passed=True,rounds=1,stage_profiling=False,previous_v25_best_pair_seconds=before,
                current_unprofiled_seconds=current,previous_speedup=old['speedup'],previous_round=old['best_round'],previous_rounds=5,
                speedup=current['gids']/current['digit'],
                seconds_saved={arm:before[arm]-current[arm] for arm in current},
                interpretation='Historical UKL v25 best of five versus new single pair; GIDS gains common GPU execution, DiGiT gains native sampling/validation optimizations; not a controlled per-change ablation')
