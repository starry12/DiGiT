"""CL full CPU RevPR/g2 admission, gated on accepted UKL feature completion."""
import hashlib
import json
import os
from pathlib import Path
from candidates.ukl_native_sampling_v10r4 import protocol as BASE
from candidates.ukl_runtime_prepare_v11r1 import protocol as UKL
from candidates.cl_safe_load_v6 import protocol as CL

ROOT=BASE.ROOT
OUT=ROOT/'results/cl_prepare_guarded_20261003_v7'
DATA_ROOT=Path('/mnt/n0/digit/cl_prepare_guarded_v7')
PYTHON=BASE.PYTHON
GIB,MIB=2**30,2**20
STAGE='graph';STAGES=('graph',)
N,E=CL.N,CL.E
READ_RATE=WRITE_RATE=GIB
APPLICATION_READ_RATE=APPLICATION_WRITE_RATE=512*MIB
RESERVE=64*GIB
EXTENT=CL.budget()['csc_arena_bytes']
HEADROOM=240*GIB-EXTENT
LOCK=ROOT/'results/overnight_ukl_cl_20261003_v1/cl_monitor.lock'
_require=BASE._require


def configure_stage(stage):
    if stage!='graph':raise ValueError('Only CL preprocessing stage is enabled')


def verify_manifest():
    UKL.verify_manifest()
    path=OUT/'manifest.json';m=json.loads(path.read_text())
    for name,h in m.items():
        p=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:
            raise RuntimeError('CL frozen dependency changed: '+name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding():
    saved=json.loads((OUT/'source_binding.json').read_text());current=CL.csc_binding()
    if saved!=current:raise RuntimeError('CL CSC metadata or identity changed')
    return current['specs']


def source_device():
    dev=binding()['indices']['identity'][0];number='%d:%d'%(os.major(dev),os.minor(dev))
    p=(Path('/dev/block')/number).resolve(strict=True)
    if not p.is_block_device():raise RuntimeError('Source block device missing')
    return str(p),number


def budget():
    return dict(arena_bytes=EXTENT,memory_max=240*GIB,memory_high=239*GIB,
                memlock=65536,host_min=304*GIB+128*MIB,own_file_cache_limit=128*MIB,
                read_rate=READ_RATE,application_read_rate=APPLICATION_READ_RATE,
                write_rate=WRITE_RATE,application_write_rate=APPLICATION_WRITE_RATE,
                runtime_seconds=43200,post_seconds=180,quiet_seconds=30,max_quiet_wait_seconds=300)


def expected_files():
    # A conservative total-output allowance for rank/hot + sparse arrays.
    return {'output_upper_bound':64*GIB}


def require_monitor_qualification(digest):
    value=json.loads((OUT/'monitor_check.json').read_text())
    if not BASE._qualification_valid(value,digest,BASE._boot_id()):
        raise RuntimeError('Current CL monitor qualification required')
    return value


def require_predecessor():
    from candidates.ukl_runtime_prepare_v11r1.finalize import collect_inputs
    path=UKL.OUT/'runtime_inputs.json'
    saved=json.loads(path.read_text())
    if saved!=collect_inputs():raise RuntimeError('UKL inputs not fully accepted')
    # Native CPU import here is avoided; controller uses system Python.
    queue=ROOT/'results/overnight_ukl_cl_20261003_v1/completion.json'
    row=json.loads(queue.read_text())
    index_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if row.get('passed') is not True or row.get('input_index_sha256')!=index_sha:
        raise RuntimeError('UKL post-completion queue must pass first')
    for name,h in row['outputs'].items():
        p=ROOT/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('Queue evidence changed')
    return dict(ukl_index_sha256=index_sha,queue_receipt_sha256=hashlib.sha256(queue.read_bytes()).hexdigest())


def _limits_valid(value,expected,device):
    return (BASE._limits_valid(value,expected,device)
            and value['io_max'][device].get('wbps')==str(WRITE_RATE))


def validate_worker_report(r):
    try:
        files=r['files'];m=r['overlay']
        expected={'revpr.f64':8*N,'hot_nodes.i64':8*(N//10),'members.i32':8*m['groups'],
                  'bases.i64':8*m['groups'],'primary.i64':8*N,'group_ptr.i64':8*(N+1),
                  'group_ids.i32':4*m['groups'],'covered.i64':16*m['groups']}
        if set(files)!=set(expected):return False
        for name,size in expected.items():
            v=files[name];p=Path(v['path'])
            if (not p.is_file() or p.is_symlink() or DATA_ROOT not in p.parents
                    or p.name!=name or v['bytes']!=size or CL.ident(p.stat())!=v['identity']
                    or not v.get('direct_io') or len(v['sha256'])!=64):return False
        return (r['passed'] is True and r['stage']=='graph' and r['manifest_sha256']==verify_manifest()
                and r['gpu_called'] is False and r['raw_ssd_access'] is False
                and r['source_revalidated'] is True and r['normal_release'] is True
                and r['ownership']['released'] is True and r['rank']['iterations']==20
                and r['rank']['nodes']==N and r['rank']['hot_nodes']==N//10
                and m['complete'] is True and m['nodes']==N and m['edges']==E
                and 0<m['groups'] and m['groups']==m['primary_groups']+m['replica_groups']
                and 0<=m['replica_groups']<=(N//5)//2 and m['storage_rows']==N+2*m['replica_groups']
                and m['second_adjacency'] is False and m['edge_sized_marker'] is False
                and _limits_valid(r['effective_limits'],budget(),source_device()[1]))
    except (KeyError,ValueError,TypeError,OSError,RuntimeError):return False


def worker_passed(state,report):return BASE._state_valid(state,budget()) and validate_worker_report(report)
