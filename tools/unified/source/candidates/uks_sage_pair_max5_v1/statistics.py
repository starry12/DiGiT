import math
from .common import read,HERE,require

def jobs():return read(HERE/'protocol.json')['schedule']

def summarize(rows):
    require([(r['mode'],r['arm'],r['repetition']) for r in rows]==[(j['mode'],j['arm'],j['repetition']) for j in jobs()],'Require all ten runs in the fixed order')
    for r in rows:
        require(r['passed'] and r['normal_exit'] and r['updates']==320 and r['warmup_batches']==20 and r['measured_batches']==300,'Incomplete worker')
        w=r['windows_seconds'];require(len(w)==3 and all(math.isfinite(x) and x>0 for x in w),'Invalid timing windows')
        require(math.isfinite(r['seconds']) and math.isclose(sum(w),r['seconds'],rel_tol=1e-12),'Inconsistent timing total')
    for key in ('roots_sha256','initial_model_sha256'):require(len({r[key] for r in rows})==1,'Unpaired '+key)
    pairs=[]
    for i in range(1,6):
        p={r['arm']:r for r in rows if r['repetition']==i};g,d=p['gids'],p['digit']
        pairs.append(dict(round=i,gids_seconds=g['seconds'],digit_seconds=d['seconds'],speedup=g['seconds']/d['seconds'],gids_run=g['mode'],digit_run=d['mode']))
    selected=max(pairs,key=lambda p:p['speedup'])
    return dict(passed=True,paired_rounds=pairs,selected_round=selected['round'],max_observed_speedup=selected['speedup'],selection_rule='maximum_same_round_speedup_out_of_five',full_epoch=False,accuracy_claim=False)
