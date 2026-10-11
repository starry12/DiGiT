"""Frozen inputs and bounded writer admission; no raw operations at import."""
import hashlib,json,os
from pathlib import Path
from candidates.ukl_runtime_prepare_v11r1 import protocol as UKL
from candidates.ukl_native_sampling_v10r4 import protocol as BASE
from candidates.ukl_feature_binding_v13 import plan as F
ROOT=UKL.ROOT;OUT=ROOT/'results/ukl_ssd_writer_20261003_v14/repair4'
DATA_ROOT=Path('/mnt/n0/digit/ukl_ssd_writer_v14r4')
QUEUE=OUT/'inputs'
PYTHON='/usr/bin/python3';GIB=2**30;MIB=2**20
READ_RATE=WRITE_RATE=GIB;APPLICATION_READ_RATE=APPLICATION_WRITE_RATE=512*MIB
STAGE='storage';STAGES=('storage',);LOCK=OUT/'controller.lock'
BINARY=OUT/'build/libukl_transport.so'
_require=BASE._require

def write(path,value,exclusive=False):
    path=Path(path);raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW
    tmp=path if exclusive else path.with_suffix('.tmp')
    fd=os.open(str(tmp),flags,0o644)
    try:
        with os.fdopen(fd,'wb',closefd=False) as f:f.write(raw);f.flush();os.fsync(fd)
    finally:os.close(fd)
    if not exclusive:os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def verify_manifest():
    from candidates.ukl_ssd_writer_v14r3.protocol import verify_manifest as previous
    previous()
    UKL.verify_manifest();p=OUT/'manifest.json';m=json.loads(p.read_text())
    for name,h in m.items():
        f=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Writer dependency changed: '+name)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def authorization():
    return F.read_metadata(QUEUE/'overwrite_authorization.json')[0]

def validate_overwrite(plan,bound,snapshot,auth):
    """Only the six frozen prior failed records may overlap these two extents."""
    from .engine import validate_regions
    clean=dict(snapshot);claimed=clean.pop('sha256')
    if F.digest(clean)!=claimed or claimed!=auth['inventory_sha256']:
        raise RuntimeError('Overwrite inventory changed')
    if (F.digest(plan)!=auth['plan_sha256'] or F.digest(bound)!=auth['binding_sha256']
            or plan['expected_device']!=auth['device'] or snapshot['expected_device']!=auth['device']):
        raise RuntimeError('Overwrite binding or device changed')
    validate_regions(bound,auth)
    files={x['path']:x for x in snapshot['files']}
    exceptions={x['path']:x for x in auth['superseded_failed_reservations']}
    if len(exceptions)!=6:raise RuntimeError('Expected exactly six superseded reservations')
    cursors=auth.get('resume_offsets',{})
    if cursors.get('gids')!=bound['arms']['gids']['region']['payload_bytes'] or not 0<cursors.get('digit',0)<bound['arms']['digit']['region']['payload_bytes']:
        raise RuntimeError('Expected complete retained GIDS and partial DiGiT')
    for name,digest in auth['resume_evidence'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Resume evidence changed')
    found=set()
    for occupied in snapshot['regions']+snapshot['reservations']:
        name=occupied['label']
        if name in exceptions:
            old=exceptions[name]
            if (files[name]['sha256']!=old['sha256'] or occupied['status']!='failed_occupied'
                    or (occupied['start'],occupied['end'])!=(old['start'],old['end'])):
                raise RuntimeError('Superseded reservation changed')
            if not any((old['start'],old['end'])==(r['guard_before'][0],r['guard_after'][1]) for r in auth['allowed_regions'].values()):
                raise RuntimeError('Superseded reservation does not exactly match approved range')
            found.add(name);continue
        for r in auth['allowed_regions'].values():
            if r['guard_before'][0]<occupied['end'] and occupied['start']<r['guard_after'][1]:
                raise RuntimeError('Overwrite overlaps protected range: '+name)
    if found!=set(exceptions):raise RuntimeError('Superseded reservations missing')

def inputs(fresh=True):
    from candidates.ukl_runtime_prepare_v11r1.finalize import collect_inputs
    a=collect_inputs();saved=json.loads((UKL.OUT/'runtime_inputs.json').read_text())
    if a!=saved:raise RuntimeError('Accepted input index changed')
    plan=F.read_metadata(QUEUE/'ssd_plan.json')[0];bound=F.read_metadata(QUEUE/'source_binding.json')[0]
    old_snapshot=F.read_metadata(QUEUE/'ssd_inventory.json')[0]
    if F.bind_sources(plan,old_snapshot,a)!=bound:raise RuntimeError('Bound source metadata changed')
    snapshot=F.read_metadata(QUEUE/'current_inventory.json')[0]
    validate_overwrite(plan,bound,F.inventory() if fresh else snapshot,authorization())
    for v in bound['arms'].values():
        s=v['source'];p=Path(s['path'])
        if p.resolve(strict=True)!=p or F.ident(p.stat())!=s['identity']:raise RuntimeError('Accepted feature identity changed')
    return plan,bound,snapshot

def binding():
    _,v,_=inputs(False)
    return {key:dict(path=s['path'],offset=0,length=s['bytes'],identity=s['identity'],sha256=s['sha256']) for key,s in [('indptr',v['arms']['gids']['source']),('indices',v['arms']['digit']['source'])]}

def source_device():
    dev=binding()['indices']['identity'][0];number='%d:%d'%(os.major(dev),os.minor(dev));p=(Path('/dev/block')/number).resolve(strict=True)
    if not p.is_block_device():raise RuntimeError('Missing source block device')
    return str(p),number

def configure_stage(stage):
    if stage!='storage':raise ValueError('Only storage writer enabled')

def budget():
    return dict(arena_bytes=8*MIB,memory_max=512*MIB,memory_high=448*MIB,memlock=64*MIB,
      host_min=64*GIB+512*MIB,own_file_cache_limit=128*MIB,read_rate=GIB,application_read_rate=512*MIB,
      write_rate=GIB,application_write_rate=512*MIB,runtime_seconds=43200,post_seconds=180,quiet_seconds=30,max_quiet_wait_seconds=300,
      global_dirty_stop_bytes=32*GIB,global_dirty_warning_bytes=2*GIB)

def expected_files():return {'small_evidence_allowance':MIB}
def require_predecessor():
    plan,bound,snapshot=inputs(True)
    return dict(plan_sha256=F.digest(plan),binding_sha256=F.digest(bound),inventory_sha256=snapshot['sha256'])
def require_monitor_qualification(digest):
    v=json.loads((OUT/'monitor_check.json').read_text())
    if not BASE._qualification_valid(v,digest,BASE._boot_id()):raise RuntimeError('Fresh writer monitor qualification required')
    return v

def _limits_valid(v,b,device):return BASE._limits_valid(v,b,device) and v['io_max'][device].get('wbps')==str(WRITE_RATE)

def state_path(r):return ROOT/'ssd_state'/('libnvm0.ukl14r4.offset'+str(r['device_offset_bytes'])+'.json')

def validate_worker_report(r):
    try:
        if not (r['passed'] and r['manifest_sha256']==verify_manifest() and r['normal_release'] and r['raw_ssd_written'] and r['gpu_called'] is False and r['raw_ssd_bound']):return False
        plan,bound,_=inputs(False)
        if r['source_binding_sha256']!=F.digest(bound):return False
        for arm,a in bound['arms'].items():
            state=F.read_metadata(state_path(a['region']))[0];v=r['arms'][arm]
            if state['status']!='verified' or state['source']!=a['source'] or state['verification']!=v or not v['sample_readback_passed'] or v['written_bytes']!=a['region']['payload_bytes'] or v['source_sha256']!=a['source']['sha256'] or v['guards_before']!=v['guards_after']:return False
            if v['overwrite_authorization_sha256']!=F.digest(authorization()) or v['blank_scan_performed'] is not False:return False
            cursor=authorization()['resume_offsets'][arm]
            if v['reused_bytes']!=cursor or v['retained_verified_bytes']!=cursor or v['new_written_bytes']!=a['region']['payload_bytes']-cursor:return False
            if arm=='gids' and not v['full_readback_passed']:return False
        return _limits_valid(r['effective_limits'],budget(),source_device()[1])
    except (OSError,ValueError,KeyError,TypeError,RuntimeError):return False

def worker_passed(state,report):return BASE._state_valid(state,budget()) and validate_worker_report(report)
