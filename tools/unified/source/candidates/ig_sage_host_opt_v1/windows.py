"""Exact batch boundaries and independent bounded-window accounting checks."""
import math
from candidates.ig_sage_host_opt_v1.common import require

def root_slices(p,smoke=False):
    require(p['epochs'] is None and p['validation'] is None and p['validation_calls']==p['test_calls']==0 and not p['epoch_time_claim'],'Evaluation/epoch claim forbidden')
    require((p['warmup_batches'],p['measurement_windows'],p['window_batches'],p['measured_batches'],p['total_train_batches'])==(20,3,100,300,320),'Wrong fixed training extent')
    if smoke:return [('training',0,p['smoke_train_batches'],p['smoke_train_batches'])]
    return [('warmup',0,20,20),('training',20,320,100)]

def check_windows(windows,batches,width,seconds):
    require(len(windows)==math.ceil(batches/width),'Missing timing windows')
    for j,w in enumerate(windows):
        lo=j*width;hi=min(batches,lo+width)
        require((w['index'],w['batch_start'],w['batch_end'],w['batches'])==(j,lo,hi,hi-lo),'Missing/overlapping/reordered batches')
        require(math.isfinite(w['seconds']) and w['seconds']>0,'Invalid timing window')
        require(0<=w['group_edges']<=w['outer_edges'],'Invalid window group coverage')
    require(math.isclose(seconds,sum(w['seconds'] for w in windows),rel_tol=1e-12),'Timing total differs')
    return True
