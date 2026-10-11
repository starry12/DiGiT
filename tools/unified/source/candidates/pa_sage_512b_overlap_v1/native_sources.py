"""Explicit, independently hashed native build inputs; no runtime/GPU imports."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'candidates/io_accounting_v1'
PY = '/home/embed/miniconda3/envs/gids/bin/python'
BINARY = HERE / 'runtime/BAM_Feature_Store/BAM_Feature_Store.so'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def native_sources():
    files = {str(p.relative_to(ROOT)): sha(p) for p in (HERE / 'native').rglob('*') if p.is_file()}
    # Keep untouched external headers bound to the frozen io-accounting tree.
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    for rel, digest in manifest['files'].items():
        if rel.startswith('native/bam/include/'):
            path = PARENT / rel
            if sha(path) != digest:
                raise RuntimeError('Frozen native include changed: ' + str(path))
            files[str(path.relative_to(ROOT))] = digest
    for path in (ROOT / 'bam/build/lib/libnvm.so', HERE / 'build.py', HERE / 'native_sources.py'):
        files[str(path.relative_to(ROOT))] = sha(path)
    return files
