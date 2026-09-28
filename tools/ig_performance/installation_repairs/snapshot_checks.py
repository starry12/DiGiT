"""Root-owned code and original read-only mounted inputs have distinct ownership contracts."""
import hashlib,os
from pathlib import Path

def check_file(path,digest,source=None):
    path=Path(path);s=path.stat()
    assert not path.is_symlink(),'Symlink in snapshot: '+str(path)
    if source is None:
        assert s.st_uid==0 and not s.st_mode&0o022,'Untrusted code: '+str(path)
    else:
        assert os.path.samefile(path,source),'Mounted input source mismatch: '+str(path)
        assert os.statvfs(path).f_flag&os.ST_RDONLY,'Mounted input is writable: '+str(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Snapshot content changed: '+str(path)

def verify_snapshot(root,files,mounts):
    root=Path(root);active=[]
    for m in mounts:
        dst=Path(m['destination']);src=Path(m['source']);dst.relative_to(root)
        if not dst.exists():continue # Newly added provenance mount is installed after this check.
        assert not dst.is_symlink(),'Symlink mount target'
        assert os.path.samefile(dst,src),'Unexpected mount source: '+str(dst)
        assert os.statvfs(dst).f_flag&os.ST_RDONLY,'Writable mount: '+str(dst)
        active.append((dst,src))
    counts=dict(code=0,mounted_inputs=0)
    for name,digest in files.items():
        rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
        p=root/rel;source=None
        for dst,src in active:
            if dst==p or dst in p.parents:source=src/p.relative_to(dst);break
        check_file(p,digest,source)
        counts['mounted_inputs' if source is not None else 'code']+=1
    return counts
