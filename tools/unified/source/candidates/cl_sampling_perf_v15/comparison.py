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
    old=json.loads((P.ROOT/'results/cl_sampling_opt_20261005_v14r2/queue_status.json').read_text())['pair']
    before={r['arm']:r['seconds'] for r in old['runs']}
    return dict(passed=True,rounds=1,stage_profiling=False,previous_profiled_seconds=before,
                current_unprofiled_seconds=current,previous_speedup=old['speedup'],
                speedup=current['gids']/current['digit'],
                seconds_saved={arm:before[arm]-current[arm] for arm in current},
                interpretation='Separate runs; timing-mode difference may also include machine variability')
