import math
from .common import config,verify
from candidates.unified_training_v2.config import digest,MODES
from candidates.unified_training_v2.accounting import validate_counts

def validate_common(r,arm,mode,repetition):
    c=config(mode);m=MODES[mode]
    if not (r['passed'] and r['complete'] and r['dataset']=='UKL' and r['arm']==arm and r['repetition']==repetition and r['mode']==mode
        and r['config_sha256']==digest(c) and not r['accuracy_evaluated'] and r['initial']['source_sha256']==verify()
        and r['initial']['evidence_scope']=='native' and r['initial']['roots_verified_exact'] and r['initial']['gpu_resident_rows_start']==0):
        raise ValueError('Wrong/incomplete native UKL common report')
    if len(r['epochs'])!=m.epochs or r['train_root_count']!=78780147:raise ValueError('Incomplete UKL epochs/split')
    updates=0
    for i,e in enumerate(r['epochs'],1):
        total=320 if m.measured_batches else (78780147+1023)//1024
        roots=320*1024 if m.measured_batches else 78780147
        if (e['epoch']!=i or e['batches']!=total or e['roots']!=roots or e['measured_batches']!=(300 if m.measured_batches else total)
            or e['measured_roots']!=(300*1024 if m.measured_batches else roots)
            or not math.isfinite(e['seconds']) or e['seconds']<=0 or e['auxiliary'].get('evaluated')):raise ValueError('UKL extent/tail/timing')
        validate_counts(e['counters'],512,arm);validate_counts(e['complete_counters'],512,arm);updates+=total
    if r['updates']!=updates or r['final']['updates']!=updates or not r['final']['finite'] or r['final']['outstanding_io'] or r['final']['outstanding_prefetch']:
        raise ValueError('Incomplete optimizer/drain')
    if not math.isclose(r['measured_seconds'],sum(e['seconds'] for e in r['epochs']),rel_tol=1e-12):raise ValueError('Timing total')
    return True
