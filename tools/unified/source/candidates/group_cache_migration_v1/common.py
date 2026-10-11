"""Small receipts and immutable source manifests for the isolated migration."""
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path('/home/embed/digit')
HERE=ROOT/'candidates/group_cache_migration_v1'
OUT=ROOT/'results/group_cache_migration_v1'
DATA=Path('/mnt/n0/digit/group_cache_migration_v1')
PY='/home/embed/miniconda3/envs/gids/bin/python'
def require(ok, message):
    if not ok: raise RuntimeError(message)
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def identity(path):
    p=Path(path);require(not p.is_symlink(),'Unexpected symlink: '+str(p));s=p.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def stable(dataset):
    sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
    import stable_identity
    stable_identity.install(dataset)
    return stable_identity
def verify():
    m=read(HERE/'manifest.json')
    for rel,h in m['files'].items():require(sha(ROOT/rel)==h,'Source changed: '+rel)
    return sha(HERE/'manifest.json')
