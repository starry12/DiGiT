"""Select and reserve one idle L40 among physical GPUs 0..3 for the whole run."""
import contextlib,csv,fcntl,io,json,os,stat,subprocess,time
from pathlib import Path

MIN_FREE_MIB=40*1024
MAX_UTILIZATION=5
LOCK_ROOT=Path('/run/digit-ae-selfservice')
def query():
    def call(args):
        return subprocess.check_output(['/usr/bin/nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=30)
    raw=call(['--query-gpu=index,uuid,name,memory.free,utilization.gpu,memory.used'])
    processes=call(['--query-compute-apps=gpu_uuid,pid'])
    busy=set()
    for row in csv.reader(io.StringIO(processes)):
        if not row or not ''.join(row).strip():continue
        if len(row)!=2 or not row[0].strip().startswith('GPU-') or not row[1].strip().isdigit():
            raise RuntimeError('Cannot establish GPU process occupancy')
        busy.add(row[0].strip())
    cards=[]
    for row in csv.reader(io.StringIO(raw)):
        if not row or not ''.join(row).strip():continue
        if len(row)!=6:raise RuntimeError('Invalid GPU inventory')
        idx,uid,name,free,util,used=[x.strip() for x in row]
        if not idx.isdigit() or not free.isdigit() or not util.isdigit() or not used.isdigit():continue
        cards.append(dict(index=int(idx),uuid=uid,name=name,free_mib=int(free),used_mib=int(used),utilization=int(util),compute_busy=uid in busy))
    if len({c['uuid'] for c in cards})!=len(cards) or len({c['index'] for c in cards})!=len(cards):
        raise RuntimeError('Ambiguous GPU inventory')
    return cards
def eligible(c,max_used_mib=1023):
    return c['index'] in range(4) and c['name']=='NVIDIA L40' and c['uuid'].startswith('GPU-') and c['free_mib']>=MIN_FREE_MIB and c['utilization']<=MAX_UTILIZATION and not c['compute_busy'] and c['used_mib']<=max_used_mib
def candidates(first,second,max_used_mib=1023):
    stable={(c['index'],c['uuid']) for c in first if eligible(c,max_used_mib)}
    return sorted((c for c in second if eligible(c,max_used_mib) and (c['index'],c['uuid']) in stable),key=lambda c:(-c['free_mib'],c['index']))
@contextlib.contextmanager
def reserve(query_fn=query,sleep_fn=time.sleep,lock_root=LOCK_ROOT,max_used_mib=1023):
    first=query_fn();sleep_fn(1);second=query_fn();chosen=None;fd=None
    for c in candidates(first,second,max_used_mib):
        path=lock_root/('uks-gpu-'+c['uuid']+'.lock')
        opened=os.open(str(path),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            s=os.fstat(opened)
            if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.geteuid() or s.st_mode&0o022:raise RuntimeError('Untrusted GPU lock')
            try:fcntl.flock(opened,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            fresh=next((x for x in query_fn() if x['uuid']==c['uuid']),None)
            if not fresh or not eligible(fresh,max_used_mib) or fresh['index']!=c['index']:continue
            chosen=dict(index=c['index'],uuid=c['uuid'],name=c['name']);fd=opened;opened=None;break
        finally:
            if opened is not None:os.close(opened)
    if chosen is None:raise RuntimeError('No idle GPU among 0,1,2,3 (need 40 GiB free, <=5% utilization, low memory use and no compute process)')
    try:yield dict(selected=chosen,first_sample=first,second_sample=second,selected_unix=time.time())
    finally:os.close(fd)
def assignment():
    value=json.loads(os.environ['DIGIT_AE_GPU_ASSIGNMENT'])
    if set(value)!=set(('index','uuid','name')) or type(value['index']) is not int or value['index'] not in range(4) or not isinstance(value['uuid'],str) or not value['uuid'].startswith('GPU-') or value['name']!='NVIDIA L40':raise RuntimeError('Invalid GPU assignment')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(value['index']):raise RuntimeError('CUDA visibility differs from selected GPU')
    return value

def transport():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text())
    import hashlib
    for name,digest in manifest['files'].items():
        path=root/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('GPU transport changed: '+name)
    return hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()

def install_assignment(selection):
    value=selection['selected']
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=str(value['index'])
    os.environ['DIGIT_AE_GPU_ASSIGNMENT']=json.dumps(value,sort_keys=True)
    assignment()

def propagated():
    if 'DIGIT_AE_GPU_ASSIGNMENT' not in os.environ:return {}
    assignment()
    return {k:os.environ[k] for k in ('CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','DIGIT_AE_GPU_ASSIGNMENT')}

def activate(stack,max_used_mib=1023):
    selection=stack.enter_context(reserve(max_used_mib=max_used_mib))
    install_assignment(selection)
    return dict(selection,transport_sha256=transport())

def idle(max_used_mib=1023):
    value=assignment()
    current=next((c for c in query() if c['uuid']==value['uuid']),None)
    if not current or current['index']!=value['index'] or not eligible(current,max_used_mib):
        raise RuntimeError('Selected GPU is no longer idle; request will not switch GPUs')
    return current

def verify_monitor(folder):
    value=assignment();p=Path(folder)
    summary=json.loads((p/'summary.json').read_text())
    if str(summary.get('gpu'))!=str(value['index']):raise RuntimeError('Monitor used another GPU')
    if 'physical_gpu_uuid' in summary and summary['physical_gpu_uuid']!=value['uuid']:
        raise RuntimeError('Monitor GPU UUID differs')

def ig_admission():
    from ae.common import host
    available=host()
    if available<320*2**30:raise RuntimeError('IG requires 320 GiB available host memory')
    current=idle()
    return dict(gpu_uuid=current['uuid'],gpu=current['index'],gpu_used_mib=current['used_mib'],
                host_available_bytes=available,checked_unix=time.time(),transport_sha256=transport())
