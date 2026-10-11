"""Lightweight project-local utilities; no GPU/framework imports."""
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(path):return json.loads(Path(path).read_text())
def cfg():return read(HERE/'protocol.json')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    os.replace(temp,path)
def source_root():return Path(os.environ.get('DIGIT_UKS_SOURCE',cfg()['paths']['source_default'])).resolve()
def identity(path):
    s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def available_host():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Missing MemAvailable')
def output_path(path):
    path=Path(path);path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    if ROOT/'results' not in path.parents:raise ValueError('New reports must be under this project results/')
    if path.exists():raise ValueError('Preserve existing output; choose a new report path')
    return path

def data_root():
    value=os.environ.get('DIGIT_UKS_DATA_ROOT',cfg()['paths']['data'])
    p=Path(value).expanduser()
    if not p.is_absolute():raise ValueError('DIGIT_UKS_DATA_ROOT must be absolute')
    p=p.resolve();allowed=[Path('/mnt/n0').resolve(),Path('/mnt/n3').resolve(),(ROOT/'data').resolve()]
    if not any(base in p.parents for base in allowed):raise ValueError('Choose a dedicated data subdirectory under /mnt/n0, /mnt/n3, or project data/')
    source=source_root()
    if p==source or source in p.parents or p in source.parents:raise ValueError('Generated data must be separate from original graph sources')
    return p

def nearest_existing(path):
    p=Path(path)
    while not p.exists():
        if p==p.parent:raise ValueError('No existing storage ancestor')
        p=p.parent
    return p

def storage_snapshot():
    p=data_root();parent=nearest_existing(p);v=os.statvfs(parent)
    return dict(data_root=str(p),existing_ancestor=str(parent),filesystem_device=parent.stat().st_dev,free_bytes=v.f_bavail*v.f_frsize,total_bytes=v.f_blocks*v.f_frsize,directory_created=False,space_is_reservation=False)

def receipt_path(path):
    path=Path(path)
    return str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)
