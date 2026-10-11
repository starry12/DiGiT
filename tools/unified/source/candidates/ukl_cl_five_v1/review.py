"""Require five accepted same-round pairs; keep every measured run."""
import hashlib,json,math
from pathlib import Path

def review(series):
    if not (series['passed'] and series['rounds_complete']==5 and series['runs_complete']==10
            and series['profiling_run'] is False and series['performance_run'] is True):
        raise RuntimeError('Five complete unprofiled pairs required')
    pairs=series['pairs'];runs=series['runs']
    if len(pairs)!=5 or len(runs)!=10 or [p['round'] for p in pairs]!=list(range(1,6)):
        raise RuntimeError('Missing or duplicate rounds')
    seen=set();accepted=[]
    for pair in pairs:
        expected=series['run_id']+'_r'+str(pair['round']);values={}
        if pair['pair_id']!=expected:raise RuntimeError('Pair identity mismatch')
        for arm in ('gids','digit'):
            run=pair[arm];path=Path(run['path']);raw=path.read_bytes();a=json.loads(raw);w=a['worker']
            if str(path) in seen:raise RuntimeError('Reused acceptance')
            seen.add(str(path))
            if not(run in runs and run['arm']==arm and run['pair_id']==expected
                   and hashlib.sha256(raw).hexdigest()==run['sha256'] and a['passed']
                   and a['pair_id']==expected and a['stage']==arm
                   and a['selected_gpu']==series['selected_gpu'] and w['selected_gpu']==series['selected_gpu']
                   and w['pair_id']==expected and a['gpu_released'] and a['kernel_monitor_ok'] and a['post_guard_passed']
                   and a['worker_state']['Result']=='success' and a['worker_state']['ExecMainStatus']=='0'
                   and w['updates']==320 and w['timing_protocol']['stage_profiling'] is False
                   and w['execution_optimizations']==dict(reused_sampling_buffers=True,gpu_postprocessing=True)
                   and math.isfinite(w['seconds']) and w['seconds']>0 and w['seconds']==run['seconds']):
                raise RuntimeError('Invalid accepted performance arm')
            values[arm]=w['seconds']
        ratio=values['gids']/values['digit']
        if not math.isfinite(ratio) or pair['speedup']!=ratio:raise RuntimeError('Invalid paired speedup')
        accepted.append(dict(round=pair['round'],pair_id=expected,**values,speedup=ratio))
    best=max(accepted,key=lambda p:p['speedup'])
    if series['best_round']!=best['round'] or series['speedup']!=best['speedup']:
        raise RuntimeError('Best-pair selection mismatch')
    return dict(passed=True,rounds=5,stage_profiling=False,best_round=best['round'],speedup=best['speedup'],
                selected_pair=best,pairs=accepted,selection='maximum same-round GIDS / DiGiT; all five pairs retained')
