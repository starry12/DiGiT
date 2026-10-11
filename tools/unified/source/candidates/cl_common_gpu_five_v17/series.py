"""Five accepted performance pairs; preserve all evidence and stop on failure."""
import json
import math
import os
import time


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(str(tmp), str(path))


def summarize(pairs, rounds=5):
    if rounds != 5 or len(pairs) != rounds or [p['round'] for p in pairs] != list(range(1, rounds+1)):
        raise ValueError('Exactly the configured number of complete ordered pairs required')
    for p in pairs:
        for arm in ('gids', 'digit'):
            r = p[arm]
            if r['arm'] != arm or not math.isfinite(r['seconds']) or r['seconds'] <= 0:
                raise ValueError('Invalid accepted timing')
            if r['pair_id'] != p['pair_id']:
                raise ValueError('Cross-round timing is forbidden')
        p['speedup'] = p['gids']['seconds'] / p['digit']['seconds']
        if not math.isfinite(p['speedup']):raise ValueError('Nonfinite paired speedup')
    best = max(pairs, key=lambda p: p['speedup'])
    return dict(passed=True, rounds_complete=rounds, runs_complete=2*rounds,
                warmup_batches=20, measured_batches=300,
                selection='maximum of five accepted same-round GIDS_seconds / DiGiT_seconds',
                best_round=best['round'], speedup=best['speedup'], pairs=pairs,
                performance_run=True, profiling_run=False, accuracy_evaluated=False,
                implementations={'gids':'GIDS + reusable buffers + GPU postprocessing','digit':'DiGiT v14 optimized sampler + GPU postprocessing'})


def run_series(out, run_id, gpu, run_one, rounds=5):
    if rounds != 5:raise ValueError('Exactly five performance rounds required')
    pairs = []; runs = []; current = None
    path = out / ('series_'+run_id+'.json')

    def record(state, **extra):
        value = dict(run_id=run_id, selected_gpu=gpu, state=state, time=time.time(),
                     stage=current, runs_complete=len(runs), rounds_complete=len(pairs),
                     passed=False, runs=runs, pairs=pairs, **extra)
        write(path, value); write(out/'latest_status.json', value)
        return value

    record('starting')
    try:
        for number in range(1, rounds+1):
            pair = dict(round=number, pair_id=run_id+'_r'+str(number))
            for arm in ('gids', 'digit'):
                current = 'round%d_%s' % (number, arm); record('running')
                result = run_one(number, arm)
                if result['arm'] != arm or result['pair_id'] != pair['pair_id']:
                    raise RuntimeError('Returned acceptance belongs to a different arm or pair')
                pair[arm] = result; runs.append(result)
                record('running')
            pair['speedup'] = pair['gids']['seconds'] / pair['digit']['seconds']
            pairs.append(pair); record('running')
        report = summarize(pairs,rounds)
        current = 'complete'
        value = record('complete')
        value.update(report); write(path, value); write(out/'latest_status.json', value)
        write(out/('summary_'+run_id+'.json'), value)
        print(json.dumps(dict(passed=True, output=str(path), speedup=value['speedup'], best_round=value['best_round'])), flush=True)
        return value
    except BaseException as error:
        record('failed', error_type=type(error).__name__, error=str(error))
        raise
