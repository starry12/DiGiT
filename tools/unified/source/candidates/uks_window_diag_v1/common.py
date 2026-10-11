import fcntl
import json
import os
from pathlib import Path
from candidates.uks_full_epoch_v1.common import ROOT,PY,require,sha,write,inputs,load_roots,describe,window_ranges

HERE=ROOT/'candidates/uks_window_diag_v1'
OUT=ROOT/'results/uks_window_diag_20261011_v1'
PARENT_SHA='8b6c511f2c115a8faa74176ee5dc29b3dafe701cd5cfbccb90bdd6ac1edbadc1'
LOCKS=('/tmp/digit-pa-sage-experiment.lock','/tmp/digit-pa-bidir-controller.lock',
       '/tmp/digit-pa-sage-libnvm0.lock',str(ROOT/'ssd_state/libnvm0.prepare.lock'),
       '/run/digit-ae-selfservice/exclusive.lock')

def verify():
    from candidates.uks_full_epoch_v1.common import verify as parent
    require(parent()==PARENT_SHA,'Latest full-epoch implementation changed')
    m=json.loads((HERE/'manifest.json').read_text())
    for name,digest in m['files'].items():require(sha(HERE/name)==digest,'Diagnostic source changed: '+name)
    return sha(HERE/'manifest.json')

def validate_inherited_locks(text):
    fds=[int(x) for x in text.split(',')]
    require(len(fds)==len(LOCKS) and len(set(fds))==len(fds),'Wrong shared lock descriptors')
    for fd,path in zip(fds,LOCKS):
        require(not Path(path).is_symlink() and os.path.samefile('/proc/self/fd/'+str(fd),path),'Lock identity changed')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)

def stage_totals(r):
    totals={};calls={}
    for w in r['windows']:
        if w['phase']!='training':continue
        for k,v in w['stages'].items():
            totals[k]=totals.get(k,0.)+v['seconds'];calls[k]=calls.get(k,0)+v['calls']
    keys={'root_transfer','sampling','address_annotation','feature_fetch','label_transfer','forward','loss','zero_grad','backward','adam'}
    require(set(totals)==keys and all(v==300 for v in calls.values()),'Incomplete stage diagnostics')
    return dict(seconds=r['measured_seconds'],stages=totals,calls=calls,
                model_seconds=sum(totals[k] for k in ('forward','loss','zero_grad','backward','adam')),
                outside_stages_seconds=r['measured_seconds']-sum(totals.values()))

def aggregate(reports):
    required={'gids','digit'}
    require(required<=set(reports),'Incomplete performance pair')
    ref=reports['gids']
    for name,r in reports.items():
        require(r['passed'] and r['updates']==320 and r['measured_batches']==300 and r['warmup_batches']==20,'Incomplete 20+300')
        require(r['profile_mode']==('stages' if name.startswith('stages_') else 'off'),'Mixed timing modes')
        require(r['training']['reconciled'] and r['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'Counters or cold-cache gate')
        for k in ('roots_sha256','initial_model_sha256','source_sha256'):require(r[k]==ref[k],'Unmatched '+k)
        for k in ('cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes'):require(r['cache'][k]==ref['cache'][k],'Unequal cache capacity')
    speedup=ref['measured_seconds']/reports['digit']['measured_seconds']
    value=dict(passed=True,speedup=speedup,aggregation='one matched pair; no max selection or window removal',
               warmup_batches=20,measured_batches=300,accuracy_evaluated=False,
               arms={arm:describe(reports[arm]['training'],reports[arm]['measured_seconds']) for arm in ('gids','digit')},
               diagnostics_required=speedup<=1,diagnostics_complete=False)
    if {'stages_gids','stages_digit'}<=set(reports):
        for arm in ('gids','digit'):
            diag=reports['stages_'+arm];normal=reports[arm]
            require(diag['shapes']==normal['shapes'] and diag['hot_file_sha256']==normal['hot_file_sha256'],'Diagnostic workload differs')
        diag={arm:stage_totals(reports['stages_'+arm]) for arm in ('gids','digit')}
        delta={k:diag['digit']['stages'][k]-diag['gids']['stages'][k] for k in diag['gids']['stages']}
        value.update(diagnostics_complete=True,diagnostics=diag,stage_deltas_digit_minus_gids=delta,
                     largest_extra_stage=max(delta,key=delta.get),
                     diagnostic_limit='Synchronized stages alter overlap; stage deltas locate overhead but do not replace unprofiled performance.')
    return value
