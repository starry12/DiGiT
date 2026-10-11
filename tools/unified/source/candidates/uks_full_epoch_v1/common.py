"""Bound full BFS epoch, existing hot sets and native policy comparison."""
import json
import math
import numpy as np
from candidates.cache_epoch_v1.common import ROOT,PY,require,sha,write
from candidates.cache_epoch_v1.core import batches,validate_order,digest_array
HERE=ROOT/'candidates/uks_full_epoch_v1'
OUT=ROOT/'results/uks_full_epoch_20261010_v1'
SOURCE_RUN=ROOT/'results/cache_epoch_20261010_v1/runs/20261010-145329-555883'

def schedule():return ['gids','digit']

def window_ranges(count):
    require(count>20,'Need full training after warmup')
    return [(0,20)]+[(lo,min(count,lo+100)) for lo in range(20,count,100)]

def verify():
    from candidates.cache_epoch_v1.common import verify as parent_verify
    m=json.loads((HERE/'manifest.json').read_text())
    require(parent_verify()==m['parent_source'],'Parent changed')
    for path,digest in m['files'].items():require(sha(ROOT/path)==digest,'Changed source/evidence: '+path)
    return sha(HERE/'manifest.json')

def inputs():
    data=json.loads((HERE/'inputs.json').read_text())
    for desc in [*data['hot_sets'].values(),data['canonical'],data['training']]:
        require(sha(desc['path'])==desc['sha256'],'Input changed: '+desc['path'])
    return data

def load_roots(arm):
    require(arm in schedule(),'Wrong arm');d=inputs()
    canonical=np.load(d['canonical']['path']);train=np.load(d['training']['path'])
    validate_order(canonical,train,d['nodes'])
    roots=np.concatenate(list(batches(canonical,epoch=0)))
    require(digest_array(roots)==d['roots_sha256'],'Full BFS order changed')
    require(len(roots)==d['train_count'],'Incomplete epoch')
    return roots

def describe(region,seconds):
    s=region['serving'];d=region['device'];size=region['request_sizes']
    return dict(seconds=seconds,serving=s,ssd=dict(
        rows=s['ssd_served_rows'],primary_commands=d['primary_commands'],
        primary_bytes=d['primary_bytes'],completed_commands=d['completed_commands'],
        completed_bytes=d['completed_bytes'],replay_commands=d['replay_commands'],
        replay_bytes=d['replay_bytes'],active_seconds=d['active_seconds'],
        request_sizes=size,grouped_ssd_row_fraction=2*size['primary_2k']/s['ssd_served_rows'] if s['ssd_served_rows'] else 0.),
        gpu_cache=region['raw_allocator_cache'])

def aggregate(reports):
    require(set(reports)==set(schedule()),'Incomplete pair')
    a,b=(reports[k] for k in schedule())
    for r in (a,b):
        require(r['passed'] and r['root_coverage_exact'] and r['updates']==math.ceil(r['examples']/1024),'Incomplete roots')
        require(r['warmup_batches']==20 and r['measured_batches']==r['updates']-20,'Wrong windows')
        require(r['tail_size']==(r['examples']%1024 or 1024),'Wrong tail')
        require(r['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'GPU not cold')
        require(r['complete_region']['reconciled'] and r['training']['reconciled'],'Invalid counters')
        require(r['full_epoch_training'] and not r['accuracy_evaluated'],'Wrong scope')
    for k in ('roots_sha256','initial_model_sha256','examples','updates','source_sha256'):
        require(a[k]==b[k],'Unmatched '+k)
    for k in ('cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes'):
        require(a['cache'][k]==b['cache'][k],'Unmatched cache '+k)
    return dict(passed=True,complete=True,epochs_per_arm=1,examples=a['examples'],updates=a['updates'],
        tail_size=a['tail_size'],matched_roots=True,matched_initial_model=True,
        full_epoch_speedup=a['seconds']/b['seconds'],
        post_warmup_speedup=a['post_warmup_seconds']/b['post_warmup_seconds'],
        arms={name:dict(full_epoch=describe(r['complete_region'],r['seconds']),
            post_warmup=describe(r['training'],r['post_warmup_seconds']),
            setup_seconds=r['setup_seconds'],epoch_wall_seconds=r['epoch_wall_seconds'],cache=r['cache'],
            warmup_seconds=r['warmup_seconds'],initial_model_sha256=r['initial_model_sha256']) for name,r in reports.items()},
        limits=['One pair, not a stability claim.','Full epoch includes the initial 20 batches; post-warmup is a separate subset.',
                'Setup/cache preload/profile time excluded. Existing DiGiT full-epoch hot set reused.',
                '100-batch timing windows with counters/progress outside timed windows, same as prior bounded test.',
                'Synthetic features/labels; no accuracy evaluation.'],accuracy_evaluated=False)
