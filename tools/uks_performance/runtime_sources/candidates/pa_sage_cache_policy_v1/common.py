import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/pa_sage_cache_policy_20260925_v1'
REFERENCE = ROOT / 'results/pa_sage_layout_shared_resume_20260925_v3/protocols/real_g2_r20.json'
ARMS = ('degree', 'revpr', 'freq', 'digit')
LABELS = dict(degree='Degree-TopK', revpr='RevPR-TopK', freq='Freq-TopK', digit='DiGiT')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify():
    manifest = read(HERE / 'manifest.json')
    for name, expected in manifest['files'].items():
        require(sha(HERE / name) == expected, 'Candidate changed: ' + name)
    for name, expected in manifest['dependencies'].items():
        require(sha(ROOT / name) == expected, 'Dependency changed: ' + name)
    return sha(HERE / 'manifest.json')
