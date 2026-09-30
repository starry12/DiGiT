import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def verify_release():
    from ae.common import verify_release as base
    manifest=json.loads((HERE/'manifest.json').read_text())
    if base()!=manifest['base_release_sha256']:raise RuntimeError('Frozen base changed')
    for rel,expected in manifest['files'].items():
        if sha(HERE/rel)!=expected:raise RuntimeError('I/O candidate changed: '+rel)
    for rel,expected in manifest['external_files'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError('I/O dependency changed: '+rel)
    return sha(HERE/'manifest.json')
