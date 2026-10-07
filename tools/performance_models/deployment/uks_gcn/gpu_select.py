"""Select and reserve one idle L40 among physical GPUs 0..3 for the whole run."""
import contextlib,csv,fcntl,io,json,os,stat,subprocess,time
from pathlib import Path

MIN_FREE_MIB=40*1024
MAX_UTILIZATION=5
LOCK_ROOT=Path('/run/digit-ae-selfservice')
def query():
    def call(args):
        return subprocess.check_output(['/usr/bin/nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=30)
    raw=call(['--query-gpu=index,uuid,name,memory.free,utilization.gpu'])
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
        if len(row)!=5:raise RuntimeError('Invalid GPU inventory')
        idx,uid,name,free,util=[x.strip() for x in row]
        if not idx.isdigit() or not free.isdigit() or not util.isdigit():continue
        cards.append(dict(index=int(idx),uuid=uid,name=name,free_mib=int(free),utilization=int(util),compute_busy=uid in busy))
    if len({c['uuid'] for c in cards})!=len(cards) or len({c['index'] for c in cards})!=len(cards):
        raise RuntimeError('Ambiguous GPU inventory')
    return cards
def eligible(c):
    return c['index'] in range(4) and c['name']=='NVIDIA L40' and c['uuid'].startswith('GPU-') and c['free_mib']>=MIN_FREE_MIB and c['utilization']<=MAX_UTILIZATION and not c['compute_busy']
def candidates(first,second):
    stable={(c['index'],c['uuid']) for c in first if eligible(c)}
    return sorted((c for c in second if eligible(c) and (c['index'],c['uuid']) in stable),key=lambda c:(-c['free_mib'],c['index']))
@contextlib.contextmanager
def reserve(query_fn=query,sleep_fn=time.sleep,lock_root=LOCK_ROOT):
    first=query_fn();sleep_fn(1);second=query_fn();chosen=None;fd=None
    for c in candidates(first,second):
        path=lock_root/('uks-gpu-'+c['uuid']+'.lock')
        opened=os.open(str(path),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            s=os.fstat(opened)
            if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.geteuid() or s.st_mode&0o022:raise RuntimeError('Untrusted GPU lock')
            try:fcntl.flock(opened,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            fresh=next((x for x in query_fn() if x['uuid']==c['uuid']),None)
            if not fresh or not eligible(fresh) or fresh['index']!=c['index']:continue
            chosen=dict(index=c['index'],uuid=c['uuid'],name=c['name']);fd=opened;opened=None;break
        finally:
            if opened is not None:os.close(opened)
    if chosen is None:raise RuntimeError('No idle GPU among 0,1,2,3 (need 40 GiB free, <=5% utilization and no compute process)')
    try:yield dict(selected=chosen,first_sample=first,second_sample=second,selected_unix=time.time())
    finally:os.close(fd)
def assignment():
    value=json.loads(os.environ['DIGIT_AE_GPU_ASSIGNMENT'])
    if set(value)!=set(('index','uuid','name')) or value['index'] not in range(4) or value['name']!='NVIDIA L40':raise RuntimeError('Invalid GPU assignment')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=value['uuid']:raise RuntimeError('CUDA visibility differs from selected GPU')
    return value
def admission():
    value=assignment();current=next((c for c in query() if c['uuid']==value['uuid']),None)
    if not current or current['index']!=value['index'] or not eligible(current):raise RuntimeError('Selected GPU is no longer idle; preserve run on the same GPU')
    from ae.common import host
    if host()<256*2**30:raise RuntimeError('Need 256 GiB host memory for UKS admission')
    return value
