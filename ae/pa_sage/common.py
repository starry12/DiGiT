"""Shared paths and contracts for the no-BFS PA/GraphSAGE reconstruction."""
import gzip
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ae.common import read, write, sha, require, host
import digit_paths

PROTOCOL = ROOT / 'configs/paper/pa_sage_random_v2.json'
DEFAULT_DATA = ROOT / 'data/papers_g2_random_v2'
DEFAULT_REPORT = ROOT / 'results/pa_sage_random_v2_preparation'


def setup_imports():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '16'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['DGLBACKEND'] = 'pytorch'
    os.environ['USE_DETERMINISTIC_ALG'] = '1'
    os.environ['DIGIT_VALIDATION_PROFILE'] = 'fast'
    os.environ['DIGIT_TRAINING_PROFILE'] = 'off'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path[:0] = [str(ROOT / 'ae/papers'), str(ROOT / 'ae/papers/runtime')]


def cfg():
    value = read(PROTOCOL)
    require(value['author_decision']['bfs_enabled'] is False, 'This entry requires the explicit no-BFS protocol')
    require(value['reconstruction_choices']['digit_train_order'] == value['reconstruction_choices']['gids_train_order'], 'Unpaired root order')
    return value


def source_config():
    return read(ROOT / 'configs/papers/protocol.json')['dataset']


def array(path):
    import numpy as np
    return np.load(path, mmap_mode='r', allow_pickle=False)


def save(path, value):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.save(f, value, allow_pickle=False)
        f.flush(); os.fsync(f.fileno())


def splits(name):
    import numpy as np
    desc = source_config()['splits'][name]
    require(sha(desc['path']) == desc['sha256'], 'Split identity changed: ' + name)
    with gzip.open(desc['path'], 'rt') as f:
        values = np.fromiter((int(line) for line in f if line.strip()), dtype=np.int64)
    require(len(values) == desc['count'] and len(np.unique(values)) == len(values), 'Bad split: ' + name)
    return values


def csc_paths():
    base = Path(digit_paths.locations()['papers_csc']) / 'csc'
    return base / 'original_indptr.npy', base / 'original_indices.npy'


def graph():
    # Preserve the established graph normalization and source EID convention.
    import dataset
    return dataset.graph_without_labels()


def progress(report, stage, **kw):
    value = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **kw)
    write(Path(report) / 'progress.json', value)
    print(__import__('json').dumps(value), flush=True)


def check_prepared(data, need_payload=True):
    data = Path(data)
    ready = read(data / 'prepared.json')
    require(ready['passed'] and ready['protocol_sha256'] == sha(PROTOCOL), 'Wrong prepared protocol')
    for relative, expected in ready['bindings'].items():
        require(sha(data / relative) == expected, 'Prepared file changed: ' + relative)
    if need_payload:
        require(ready['filesystem_payload_verified'], 'Filesystem payload not verified')
    return ready
