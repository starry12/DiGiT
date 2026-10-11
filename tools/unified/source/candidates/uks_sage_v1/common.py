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
