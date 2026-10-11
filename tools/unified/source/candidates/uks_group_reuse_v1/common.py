"""Source and input bindings for a separate, immutable adjacency overlay."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/home/embed/digit')
HERE=ROOT/'candidates/uks_group_reuse_v1'
OUT=ROOT/'results/uks_group_reuse_20261010_v1'
DATA=Path('/mnt/n0/digit/uks_prepare_20260929_v3')
DEFAULT_OUTPUT=Path('/mnt/n0/digit/uks_group_reuse_20261010_v1')
BINARY=HERE/'bin/reuse'
PY='/home/embed/miniconda3/envs/gids/bin/python'
ARRAYS=('reordered_indptr','reordered_indices')

def require(ok,why):
    if not ok:raise RuntimeError(why)

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def identity(path):
    p=Path(path);require(not p.is_symlink(),'Symlink input: '+str(p));s=p.stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def verify():
    m=read(HERE/'manifest.json')
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Candidate changed: '+name)
    for path,h in m['dependencies'].items():require(sha(path)==h,'Dependency changed: '+path)
    require(sha(BINARY)==m['builder_sha256'],'Builder changed')
    return sha(HERE/'manifest.json')

def stable():
    sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
    import stable_identity
    stable_identity.install('UKS')
    return stable_identity

def source_binding():
    # Reuse the installed UUID + remaining stat-field checks. Never accept a
    # change of inode/size/timestamps as a harmless device-number change.
    stable()
    from candidates.uks_native_v1.binding import check
    from candidates.uks_freq_bfs_retry_v1.common import check_ready
    check_ready();return check()

@contextlib.contextmanager
def exclusive():
    with contextlib.ExitStack() as stack:
        for p in (Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),
                  Path('/tmp/digit-pa-sage-libnvm0.lock'),ROOT/'ssd_state/libnvm0.prepare.lock',
                  Path('/run/digit-ae-selfservice/exclusive.lock')):
            require(p.is_file() and not p.is_symlink(),'Missing/unsafe shared lock: '+str(p))
            f=stack.enter_context(p.open('rb'))
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another experiment is active; retry preparation after it finishes') from None
        yield

def check_prepared(folder,full_hash=False):
    folder=Path(folder).resolve();r=read(folder/'prepared.json');source=verify()
    require(r['passed'] and r['source_sha256']==source and r['validation']['passed'],'Missing or stale overlay acceptance')
    require(r['raw_ssd_writes'] is False and r['feature_layout_changed'] is False,'Unexpected layout change')
    require(r['base_binding_sha256']==sha(ROOT/'results/uks_native_20260929_v1/binding.json'),'Base binding changed')
    current=source_binding();require(r['base_files']==current['files'],'Different input provenance')
    s=stable()
    for name in ARRAYS:
        p=folder/(name+'.npy');desc=r['files'][name]
        old=desc['identity'];now=identity(p)
        require(all(now[k]==old[k] for k in old if k!='device'),'Overlay changed: '+name)
        require(s.filesystem_uuid(p)==desc['filesystem_uuid'],'Overlay filesystem changed')
        if full_hash:require(sha(p)==desc['sha256'],'Overlay hash changed')
    return r
