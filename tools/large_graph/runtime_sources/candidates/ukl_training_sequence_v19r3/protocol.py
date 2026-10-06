"""Smoke scope only; immutable predecessors and effective limits are mandatory."""
import hashlib,json,math,os
from pathlib import Path
from candidates.ukl_ssd_writer_v14r4 import protocol as W
from candidates.ukl_training_native_v15r11 import protocol as T
from candidates.ukl_training_native_v15r11.protocol import require_admitted_worker
from candidates.ukl_training_native_v15r11.budget import budget as training_budget
from candidates.ukl_training_native_v15r11.storage import accepted_storage
ROOT=W.ROOT;OUT=ROOT/'results/ukl_training_sequence_20261003_v19/repair3';LOCK=OUT/'controller.lock'
UKL=W.UKL;BASE=W.BASE;GIB=W.GIB;MIB=W.MIB
PYTHON='/home/embed/miniconda3/envs/gids/bin/python'
READ_RATE=WRITE_RATE=GIB;APPLICATION_READ_RATE=512*MIB
STAGE='gids';STAGES=('gids','digit');PAIR_ID='';SELECTED=None
TRAINING_SHA='3041d2152cd7712406646ad001024b432abca4830398f87788ab25a260ffa7a1'

def small(stage=None):return (stage or STAGE).startswith('ssd_')
def arm(stage=None):return (stage or STAGE).replace('ssd_','')

def configure_stage(stage):
    global STAGE
    if stage not in STAGES:raise ValueError('Only gids and digit smoke arms')
    STAGE=stage

def verify_manifest():
    if T.verify_manifest()!=TRAINING_SHA:raise RuntimeError('Frozen production backend changed')
    p=OUT/'manifest.json'
    for rel,h in json.loads(p.read_text()).items():
        f=ROOT/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Smoke dependency changed: '+rel)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def binding():return UKL.BASE.binding()
def source_device():return UKL.BASE.source_device()
def budget(arm=None):
    stage=arm or STAGE
    b=training_budget(stage.replace('ssd_',''))
    if small(stage):b.update(host_cgroup_max=8*GIB,host_cgroup_high=7*GIB,memlock=GIB,host_min=72*GIB)
    return dict(arena_bytes=b['host_components']['graph_arena'],memory_max=b['host_cgroup_max'],memory_high=b['host_cgroup_high'],memlock=b['memlock'],host_min=b['host_min'],own_file_cache_limit=b['own_file_cache_limit'],read_rate=READ_RATE,write_rate=WRITE_RATE,application_read_rate=APPLICATION_READ_RATE,runtime_seconds=900 if small(stage) else 5400,post_seconds=30,failure_post_seconds=180,quiet_seconds=30,max_quiet_wait_seconds=300)

def require_predecessor():
    v=accepted_storage()
    life=ROOT/'results/ukl_cache_lifecycle_20261003_v18/repair1/completion_review.json'
    review=json.loads(life.read_text())
    if not review['passed'] or set(review['cases'])!={'legacy_alloc','normal','cancel_alloc','cancel_preload'} or any(not x['passed'] for x in review['cases'].values()):raise RuntimeError('Accepted lifecycle tests required')
    result=dict(storage_acceptance_sha256=v['acceptance_sha256'],source_binding_sha256=v['source_binding_sha256'],lifecycle_sha256=hashlib.sha256(life.read_bytes()).hexdigest())
    from .reuse import accepted_preloads
    result['reused_preloads']=accepted_preloads(ROOT,OUT/'reused_preloads.json',v)
    for prior in STAGES[:STAGES.index(STAGE)]:
        matches=[]
        for path in OUT.glob('stage_'+prior+'_run_*/acceptance.json'):
            a=json.loads(path.read_text())
            if a.get('pair_id')==PAIR_ID:matches.append((path,a))
        if len(matches)!=1:raise RuntimeError('Exactly one same-sequence accepted predecessor required: '+prior)
        path,a=matches[0]
        if not (a['passed'] and a['selected_gpu']==list(SELECTED) and a['kernel_monitor_ok'] and a['post_guard_passed'] and a['gpu_released'] and worker_passed(a['worker_state'],a['worker'],prior)):raise RuntimeError('Predecessor not accepted: '+prior)
        result[prior+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def ownership_passed(value,expected_arenas):
    return (isinstance(value,dict) and value.get('released') is True
            and value.get('active') is False
            and type(value.get('tracked_arenas')) is int
            and type(value.get('released_arenas')) is int
            and value['tracked_arenas']==value['released_arenas']==expected_arenas)

def validate_worker_report(r,arm=None):
    arm=arm or STAGE
    if small(arm):return validate_preload(r,arm)
    try:
        if not (r['passed'] is True and r['arm']==arm and r['mode']=='smoke' and r['updates']==r['measured_batches']==4 and r['warmup_batches']==0 and r['finite'] is True and r['accuracy_evaluated'] is False and r['graph_and_aux_released'] is True and r['process_exit_acceptance_required'] is True):return False
        if not (math.isfinite(r['seconds']) and r['seconds']>0 and r['initial_model_sha256']!=r['final_model_sha256'] and r['manifest_sha256']==verify_manifest() and r['pair_id']==PAIR_ID and r['selected_gpu']==list(SELECTED)):return False
        if r['storage_acceptance_sha256']!=accepted_storage()['acceptance_sha256']:return False
        phase=r['phase_accounting']
        if not (phase['passed'] is True and phase['graph_allocated'] is False and phase['probe_bytes_each']==4*MIB and phase['phase_ack']['phase']=='data' and phase['phase_ack']['initialization_file_limit']==GIB and phase['phase_ack']['data_file_limit']==128*MIB):return False
        if not (4*MIB<=phase['init']['file']<=GIB and 4*MIB<=phase['data']['file']<=128*MIB):return False
        if len(r['feature_value_checks'])!=4 or any(v['passed'] is not True or v['rows']!=32 or v['dim']!=128 for v in r['feature_value_checks']):return False
        if set(r['counters'])!={'whole','measured'}:return False
        for v in r['counters'].values():
            if v['reconciled'] is not True or v['feature_row_bytes']!=512 or v['device']['submitted_commands']!=v['device']['completed_commands']:return False
            if not 4096<=v['serving']['logical_requests']<=4*405504:return False
        if not ownership_passed(r.get('ownership'),2):return False
        if not r['external_cache_released'] or not r['cache']['segmented_anonymous']:return False
        io=r['cache']['preload_io']
        if io['outstanding'] or io['submitted_commands']<=0 or io['submitted_commands']!=io['completed_commands']:return False
        if r['maxrss_kib']*1024>budget(arm)['memory_max']:return False
        return r['effective_limits_verified'] is True and r['raw_ssd_writes'] is False
    except (KeyError,ValueError,TypeError,RuntimeError,OSError):return False

def worker_passed(state,report,arm=None):return BASE._state_valid(state,budget(arm)) and validate_worker_report(report,arm)

def validate_preload(r,stage):
    try:
        return (r['passed'] is True and r['mode']=='ssd_preload' and r['arm']==arm(stage) and r['stage']==stage
          and r['rows']==65536 and r['values_checked']==65536*128 and r['updates']==0
          and r['cache_released'] is True and r['graph_loaded'] is False
          and ownership_passed(r.get('ownership'),1 if arm(stage)=='digit' else 0)
          and r['io']['submitted_commands']>0 and r['io']['submitted_commands']==r['io']['completed_commands']
          and r['io']['outstanding']==0 and r['io']['completed_bytes']>0
          and r['selected_gpu']==list(SELECTED) and r['pair_id']==PAIR_ID
          and r['manifest_sha256']==verify_manifest() and r['storage_acceptance_sha256']==accepted_storage()['acceptance_sha256']
          and r['effective_limits_verified'] is True and r['raw_ssd_writes'] is False
          and r['maxrss_kib']*1024<=budget(stage)['memory_max'])
    except (KeyError,TypeError,ValueError,RuntimeError,OSError):return False
