import hashlib,json,os
from pathlib import Path
from candidates.ukl_native_sampling_v10r4 import protocol as BASE
from candidates.cl_safe_load_v6.protocol import csc_binding

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cl_night_prepare_20261004_v8'
DATA_ROOT=Path('/mnt/n0/digit/cl_night_prepare_v8')
PYTHON='/home/embed/miniconda3/envs/gids/bin/python'
GIB=2**30;MIB=2**20;RESERVE=64*GIB
READ_RATE=WRITE_RATE=GIB;APPLICATION_RATE=512*MIB
STAGES=('mapping','split','features-gids','features-digit','graph')
STAGE='mapping';LOCK=OUT/'controller.lock'
N=978408098;ROWS=1174089716
BUILD=ROOT/'results/ukl_runtime_prepare_20261002_v11/build/libprepare_cpu.so'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ident(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def metadata():return json.loads((OUT/'source_binding.json').read_text())
def configure_stage(stage):
    global STAGE
    if stage not in STAGES:raise ValueError('Unknown CPU preparation stage')
    STAGE=stage
def verify_manifest():
    p=OUT/'manifest.json'
    for name,h in json.loads(p.read_text()).items():
        f=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or f.is_symlink() or sha(f)!=h:
            raise RuntimeError('Frozen preparation dependency changed: '+name)
    return sha(p)
def binding():
    b=metadata();a=json.loads((ROOT/b['receipt']).read_text())
    if sha(ROOT/b['receipt'])!=b['receipt_sha256'] or not a['passed'] or not a['post_guard_passed'] or not a['kernel_monitor_ok']:
        raise RuntimeError('Accepted CL graph preparation changed')
    if csc_binding()!=b['csc']:raise RuntimeError('CL CSC source identity changed')
    for row in list(b['specs'].values())+[b['revpr']]:
        p=Path(row['path'])
        if p.is_symlink() or p.resolve()!=p or not p.is_file() or ident(p)!=row['identity']:
            raise RuntimeError('CL source replaced: '+str(p))
    return b['specs']
def source_device():
    dev=binding()['indices']['identity'][0];number='%d:%d'%(os.major(dev),os.minor(dev))
    p=(Path('/dev/block')/number).resolve(strict=True)
    if not p.is_block_device():raise RuntimeError('Source is not a block device')
    return str(p),number
def budget(stage=None):
    stage=stage or STAGE
    maximum={'mapping':8,'split':16,'features-gids':2,'features-digit':8,'graph':304}[stage]*GIB
    return dict(memory_max=maximum,memory_high=maximum-512*MIB,memlock=65536,
        host_min=maximum+RESERVE+128*MIB,own_file_cache_limit=128*MIB,
        read_rate=READ_RATE,write_rate=WRITE_RATE,application_read_rate=APPLICATION_RATE,
        runtime_seconds=43200,post_seconds=30,failure_post_seconds=180,
        quiet_seconds=30,max_quiet_wait_seconds=60)
def expected_files(stage=None):
    return {
        'mapping':{'storage_to_node.i32':ROWS*4},
        'split':{'train.i64':(N//10)*8,'gids_roots.i64':320*1024*8,'freq_roots.i64':100*1024*8,'gids_labels.i64':320*1024*8},
        'features-gids':{'gids_features.f32':N*512},
        'features-digit':{'digit_features.f32':ROWS*512},
        'graph':{'bfs.i64':(N//10)*8,'digit_roots.i64':320*1024*8,'digit_labels.i64':320*1024*8,
                 'freq_counts.u64':N*8,'freq_hot.i64':(N//10)*8,'gids_hot.i64':(N//10)*8}
    }[stage or STAGE]
def dependencies(stage=None):return {'features-digit':('mapping',),'graph':('split',)}.get(stage or STAGE,())
def valid_files(files,stage):
    if set(files)!=set(expected_files(stage)):return False
    for name,size in expected_files(stage).items():
        row=files[name];p=Path(row['path'])
        if (not p.is_file() or p.is_symlink() or p.resolve()!=p or DATA_ROOT not in p.parents
                or p.name!=name or row['bytes']!=size or ident(p)!=row['identity']
                or not row.get('direct_io') or len(row['sha256'])!=64):return False
    return True
def validate_worker_report(r,stage=None):
    stage=stage or STAGE
    try:
        good=(r['passed'] is True and r['stage']==stage and r['manifest_sha256']==verify_manifest()
            and r['source_revalidated'] is True and r['normal_release'] is True
            and r['gpu_called'] is False and r['raw_ssd_access'] is False
            and r['maxrss_kib']*1024<=budget(stage)['memory_max'] and valid_files(r['files'],stage))
        if not good:return False
        o=r['ownership']
        if not (o['released'] is True and o['active'] is False and o['tracked_arenas']==o['released_arenas']):return False
        if not _limits_valid(r['effective_limits'],budget(stage),source_device()[1]):return False
        c=r['checks']
        if stage=='mapping':return c['primary_roundtrip'] is True and c['all_rows_mapped'] is True
        if stage=='split':return c['train_nodes']==N//10 and c['sorted_unique'] is True
        if stage.startswith('features-'):return c['passed'] is True and c['rows_checked']>=2 and c['logical_node_consistent'] is True
        return (c['freq']['batches']==100 and c['freq']['hot_nodes']==N//10 and c['freq']['independent'] is True
                and c['bfs']['same_training_set'] is True and c['bfs']['train_nodes']==N//10)
    except (OSError,KeyError,ValueError,TypeError,RuntimeError):return False
def worker_passed(state,report):return BASE._state_valid(state,budget()) and validate_worker_report(report)
def accepted_stage(stage):
    for path in sorted(OUT.glob('stage_'+stage+'_run_*/acceptance.json'),reverse=True):
        a=json.loads(path.read_text())
        if (a.get('passed') is True and a.get('manifest_sha256')==verify_manifest()
            and a.get('kernel_monitor_ok') is True and a.get('post_guard_passed') is True
            and BASE._state_valid(a['worker_state'],budget(stage)) and validate_worker_report(a['worker'],stage)):
            expected=dict(graph_receipt_sha256=metadata()['receipt_sha256'],
                          stages={s:accepted_stage(s) for s in dependencies(stage)})
            if a['worker'].get('predecessor')!=expected or a.get('predecessor')!=expected:
                continue
            return dict(path=str(path),sha256=sha(path),files=a['worker']['files'])
    raise RuntimeError('Accepted prerequisite missing: '+stage)
def require_predecessor():
    binding();b=metadata()
    return dict(graph_receipt_sha256=b['receipt_sha256'],stages={s:accepted_stage(s) for s in dependencies()})
def _limits_valid(value,expected,device):
    return BASE._limits_valid(value,expected,device) and value['io_max'][device].get('wbps')==str(WRITE_RATE)
