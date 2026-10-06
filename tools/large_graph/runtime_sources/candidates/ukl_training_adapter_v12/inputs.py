"""Read accepted preparation metadata, never unfinished feature payloads."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def accepted_inputs():
    from candidates.ukl_runtime_prepare_v11r1 import protocol as P
    from candidates.ukl_runtime_prepare_v11r1.finalize import collect_inputs
    path = P.OUT / 'runtime_inputs.json'
    if not path.is_file():
        raise RuntimeError('Feature preparation is not yet accepted; runtime_inputs.json missing')
    value = json.loads(path.read_text())
    # Recompute from accepted receipts, not merely prepared=true in a JSON file.
    if value != collect_inputs():
        raise RuntimeError('Runtime input index differs from accepted preparation receipts')
    return value


def arm_inputs(value, arm):
    if arm not in ('gids', 'digit'):
        raise ValueError('Unknown arm')
    if (value.get('prepared') is not True or value.get('feature_dim') != 128
            or value.get('row_bytes') != 512 or value.get('classes') != 19
            or value.get('feature_dtype') != 'float32'):
        raise ValueError('UKL prepared input protocol mismatch')
    graph = value['stages']['graph']['files']
    return dict(roots=graph[arm + '_roots.i64'], labels=graph[arm + '_labels.i64'],
                hot=graph['gids_hot.i64' if arm == 'gids' else 'freq_hot.i64'],
                features=value['stages']['features-' + arm]['files'][arm + '_features.f32'],
                nodes=value['nodes'], rows=value['nodes'] if arm == 'gids' else value['storage_rows'],
                arm=arm, cpu_cache_rows=value['cpu_cache_rows'],
                gpu_cache_bytes=value['gpu_cache_bytes'], raw_ssd_bound=False)


def read_small_i64(receipt, count):
    """Only roots/labels (<= 2.5 MiB); reject large hot/feature files before open."""
    if type(count) is not int or not 0 < count <= 320 * 1024:
        raise ValueError('Small input read bound')
    if receipt.get('bytes') != count * 8:
        raise ValueError('Input byte length mismatch')
    path = Path(receipt['path'])
    if path.name.endswith('.partial') or path.resolve(strict=True) != path:
        raise ValueError('Published absolute input without symlinks required')
    expected = receipt['identity']
    def ident(s):
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if ident(os.fstat(fd)) != expected:
            raise RuntimeError('Input identity changed')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(count * 8 + 1)
        if (len(raw) != count * 8 or hashlib.sha256(raw).hexdigest() != receipt['sha256']
                or ident(os.fstat(fd)) != expected or ident(path.stat()) != expected):
            raise RuntimeError('Input changed or digest mismatch')
    finally:
        os.close(fd)
    return np.frombuffer(raw, dtype=np.int64).copy()


def load_window(arm_input):
    roots = read_small_i64(arm_input['roots'], 320 * 1024)
    labels = read_small_i64(arm_input['labels'], 320 * 1024)
    if (np.any(roots < 0) or np.any(roots >= arm_input['nodes'])
            or len(np.unique(roots)) != len(roots)
            or np.any(labels < 0) or np.any(labels >= 19)):
        raise ValueError('Window roots/labels outside protocol')
    # Labels are aligned to this arm's window; never index a window by global ID.
    return roots.reshape(320, 1024), labels.reshape(320, 1024)
