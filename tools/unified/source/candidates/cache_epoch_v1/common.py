"""Source binding and small atomic reports; no device operations at import."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/embed/digit')
HERE = ROOT/'candidates/cache_epoch_v1'
OUT = ROOT/'results/cache_epoch_20261010_v1'
PY = '/home/embed/miniconda3/envs/gids/bin/python'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(4*2**20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def verify():
    manifest = HERE/'manifest.json'
    data = json.loads(manifest.read_text())
    for rel, expected in data['files'].items():
        path = ROOT/rel
        require(not path.is_symlink() and sha(path) == expected, 'Source changed: '+rel)
    return sha(manifest)
