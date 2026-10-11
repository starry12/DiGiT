"""Validate frozen implementation files without touching graph or SSD data."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ukl_cl_five_20261006_v1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(archive=False):
    manifest=OUT/'frozen_manifest.json';seal=json.loads((OUT/'seal.json').read_text())
    if digest(manifest)!=seal['manifest_sha256']:raise RuntimeError('Frozen manifest changed')
    for rel,h in json.loads(manifest.read_text()).items():
        p=ROOT/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or p.is_symlink() or digest(p)!=h:
            raise RuntimeError('Frozen dependency changed: '+rel)
    if archive and digest(OUT/'frozen_sources.tar')!=seal['archive_sha256']:
        raise RuntimeError('Frozen archive changed')
    return seal['manifest_sha256']
