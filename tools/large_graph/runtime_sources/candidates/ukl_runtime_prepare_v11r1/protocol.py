"""Three independently limited CPU stages, gated by accepted anonymous sampling."""
import hashlib
import json
import os
from pathlib import Path
from candidates.ukl_native_sampling_v10r4 import protocol as BASE

def ident(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

ROOT=BASE.ROOT
OUT=ROOT/'results/ukl_runtime_prepare_20261002_v11/repair1'
DATA_ROOT=Path('/mnt/n0/digit/ukl_runtime_prepare_v11r1')
BUILD=ROOT/'results/ukl_runtime_prepare_20261002_v11/build/libprepare_cpu.so'
GIB,MIB=BASE.GIB,BASE.MIB
PYTHON,LOCK=BASE.PYTHON,BASE.LOCK
NAMES=BASE.NAMES
STAGES=('graph','features-gids','features-digit')
STAGE='graph'
EXTENT=BASE.EXTENT
READ_RATE=GIB
APPLICATION_READ_RATE=512*MIB
WRITE_RATE=GIB
APPLICATION_WRITE_RATE=512*MIB
HEADROOM=304*GIB-EXTENT
RESERVE=64*GIB
METADATA=BASE.METADATA
N=METADATA['nodes'];ROWS=METADATA['storage_rows']
BATCH=1024;WINDOW_BATCHES=320;FREQ_BATCHES=100
REVPR_PATH=Path('/mnt/n0/digit/ukl_sparse_overlay_v3/revpr_overlay_v4/revpr/hot_nodes.i64')
REVPR_SHA='8492e1f6144f58955cef251937e145b3bd82ebd42eecde4fb9fd213b484d3878'
PREVIOUS=BASE.OUT/'stage_sampling_run_20261002_222003_1754494'

_require=BASE._require
binding=BASE.binding
source_device=BASE.source_device

def configure_stage(stage):
    global STAGE,EXTENT
    if stage not in STAGES:raise ValueError('unknown stage')
    STAGE=stage
    EXTENT=BASE.EXTENT if stage=='graph' else ((ROWS*4+4095)//4096*4096 if stage=='features-digit' else 0)

def budget():
    maximum={'graph':304*GIB,'features-gids':2*GIB,'features-digit':12*GIB}[STAGE]
    return dict(arena_bytes=EXTENT,memory_max=maximum,memory_high=maximum-512*MIB,
                memlock=65536,host_min=maximum+RESERVE+128*MIB,own_file_cache_limit=128*MIB,
                read_rate=READ_RATE,application_read_rate=APPLICATION_READ_RATE,
                write_rate=WRITE_RATE,application_write_rate=APPLICATION_WRITE_RATE,
                runtime_seconds=43200,post_seconds=180,quiet_seconds=30,max_quiet_wait_seconds=300)

def expected_files(stage=None):
    stage=stage or STAGE
    return {'train.i64':N//10*8,'gids_roots.i64':WINDOW_BATCHES*BATCH*8,
     'freq_roots.i64':FREQ_BATCHES*BATCH*8,'bfs.i64':N//10*8,
     'digit_roots.i64':WINDOW_BATCHES*BATCH*8,'freq_counts.u64':N*8,
     'freq_hot.i64':N//10*8,'storage_to_node.i32':ROWS*4,
     'gids_labels.i64':WINDOW_BATCHES*BATCH*8,'digit_labels.i64':WINDOW_BATCHES*BATCH*8,
     'gids_hot.i64':N//10*8} if stage=='graph' else {stage.replace('features-','')+'_features.f32':(N if stage=='features-gids' else ROWS)*512}

def verify_manifest():
    BASE.verify_manifest()
    path=OUT/'manifest.json';entries=json.loads(path.read_text())
    _require(bool(entries),'empty manifest')
    for name,digest in entries.items():
        relative=Path(name)
        _require(not relative.is_absolute() and '..' not in relative.parts,'invalid manifest path')
        path=ROOT/relative
        _require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'changed '+name)
    return hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()

def require_monitor_qualification(digest):
    result=json.loads((OUT/'monitor_check.json').read_text())
    _require(BASE._qualification_valid(result,digest,BASE._boot_id()),'current monitor qualification required')
    return result

def _valid_files(report,stage):
    expected=expected_files(stage);files=report.get('files',{})
    if set(files)!=set(expected):return False
    for name,size in expected.items():
        row=files[name];path=Path(row['path'])
        if (path.name!=name or path.parent.parent!=DATA_ROOT or path.is_symlink()
                or path.parent.is_symlink() or not row.get('direct_io')
                or row.get('bytes')!=size or not BASE._hash(row.get('sha256'))
                or ident(path)!=row['identity'] or path.stat().st_size!=size):return False
    return True

def require_predecessor():
    a=json.loads((PREVIOUS/'acceptance.json').read_text());w=json.loads((PREVIOUS/'worker.json').read_text())
    _require(a['passed'] and a['kernel_monitor_ok'] and a['post_guard_passed'] and a['worker']==w
             and a['manifest_sha256']==BASE.verify_manifest() and BASE.validate_worker_report(w)
             and BASE.worker_passed(a['worker_state'],w),'full graph sampling predecessor failed')
    result=dict(native_acceptance=str(PREVIOUS),native_acceptance_sha256=hashlib.sha256((PREVIOUS/'acceptance.json').read_bytes()).hexdigest())
    if STAGE!='graph':
        runs=sorted(OUT.glob('stage_graph_run_*'))
        _require(bool(runs),'graph preparation must complete first')
        r=runs[-1];accepted=json.loads((r/'acceptance.json').read_text());report=accepted['worker']
        _require(accepted['passed'] and accepted['manifest_sha256']==verify_manifest()
                 and report.get('stage')=='graph' and _valid_files(report,'graph'),'latest graph preparation not accepted')
        result.update(graph_run=str(r),graph_files=report['files'],graph_receipt_sha256=hashlib.sha256((r/'acceptance.json').read_bytes()).hexdigest())
    return result

def _limits_valid(limits,expected,device):
    return (BASE._limits_valid(limits,expected,device)
            and limits['io_max'][device].get('wbps')==str(WRITE_RATE))

def validate_worker_report(report):
    try:
        number=binding()['indices']['identity'][0];device='%d:%d'%(os.major(number),os.minor(number))
        return (report['passed'] is True and report['stage']==STAGE
            and report['gpu_called'] is False and report['raw_ssd_access'] is False
            and report['normal_release'] is True and report['source_revalidated'] is True
            and report['manifest_sha256']==verify_manifest()
            and _limits_valid(report['effective_limits'],budget(),device)
            and report['application_read_rate']==APPLICATION_READ_RATE
            and report['application_write_rate']==APPLICATION_WRITE_RATE
            and _valid_files(report,STAGE)
            and (STAGE!='graph' or (report['freq']['batches']==FREQ_BATCHES and report['freq']['seed']==23
                 and report['freq']['hot_nodes']==N//10 and report['bfs']['train_nodes']==N//10
                 and report['bfs']['same_training_set'] is True))
            and (STAGE=='graph' or report['feature_checks']['passed'] is True))
    except (KeyError,ValueError,TypeError,OSError,RuntimeError):return False

def worker_passed(state,report):
    return BASE._state_valid(state,budget()) and validate_worker_report(report)
