import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def verify():
    from candidates.group_cache_migration_v1.common import verify as parent
    parent()
    m=json.loads((HERE/'manifest.json').read_text())
    for rel,h in m['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Compact source changed: '+rel)
    return hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()
