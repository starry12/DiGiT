"""Small fixed-destination installation primitives."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record(p):
    p = Path(p)
    if p.is_symlink():
        return dict(link=os.readlink(p))
    return dict(sha256=sha(p), mode=p.stat().st_mode & 0o777) if p.exists() else None


def atomic(p, data, mode=0o644):
    p = Path(p)
    require(not p.is_symlink(), 'Symlink target: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + p.name, dir=p.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            os.fchmod(f.fileno(), mode)
        os.replace(name, p)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def invoke(args, check=True):
    return subprocess.run(list(map(str, args)), check=check, capture_output=True,
                          text=True, timeout=600)


def tree(source, dest):
    dest = Path(dest)
    require(not dest.is_symlink(), 'Symlink destination')
    dest.mkdir(parents=True, exist_ok=True)
    dest.chmod(0o755)
    for p in sorted(source.rglob('*')):
        require(not p.is_symlink(), 'Source symlink')
        t = dest / p.relative_to(source)
        require(not t.is_symlink(), 'Destination symlink')
        if p.is_dir():
            t.mkdir(exist_ok=True)
            t.chmod(0o755)
        elif t.exists():
            require(sha(t) == sha(p), 'Existing installed file differs: ' + str(t))
        else:
            atomic(t, p.read_bytes(), 0o755 if p.stat().st_mode & 0o111 else 0o644)
    for p in [dest, *dest.rglob('*')]:
        stat = p.stat()
        require(stat.st_uid == 0 and not stat.st_mode & 0o022, 'Untrusted installed file: ' + str(p))


def pointer(target):
    p = Path('/srv/digit-ae/reviewer-current')
    tmp = p.with_name('.multimodel-reviewer-' + str(os.getpid()))
    require(not tmp.exists() and not tmp.is_symlink(), 'Temporary pointer exists')
    os.symlink(str(target), tmp)
    os.replace(tmp, p)
