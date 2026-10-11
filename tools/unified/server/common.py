import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path('/home/embed/digit')
ADMIN = Path('/srv/digit-ae/admin/unified_20261011_v1')
OUT = Path('/srv/digit-ae/unified-results')
PYTHON = '/home/embed/miniconda3/envs/gids/bin/python'
DATASETS = ('PA', 'IG', 'UKS', 'UKL', 'CL')
MODES = ('short1', 'short5', 'full20')
ENTRY = ROOT / 'candidates/unified_experiments_v2/manifest.json'

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text())

def inside(path, parent):
    try:
        Path(path).relative_to(parent)
        return '..' not in Path(path).parts
    except ValueError:
        return False

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    atomic(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())

def atomic(path, data, mode=0o644):
    path = Path(path)
    require(not path.is_symlink(), 'Unexpected symlink: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.unified-')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            os.fchmod(f.fileno(), mode)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def unit(dataset, mode):
    require(dataset in DATASETS and mode in MODES, 'Unknown dataset/mode')
    return 'digit-ae-unified-' + dataset.lower() + '-' + mode + '.service'

def service(dataset, mode):
    text = subprocess.check_output(['/usr/bin/systemctl', 'show', unit(dataset, mode),
        '--property=ActiveState,SubState,Result,ExecMainStatus,InvocationID'], text=True, timeout=15)
    return dict(line.split('=', 1) for line in text.splitlines() if '=' in line)

def active(state):
    return state.get('ActiveState') in ('active', 'activating', 'deactivating', 'reloading')

def output(dataset, mode, name):
    parent = OUT / dataset / mode / 'runs'
    p = Path(name)
    require(p.parent == parent and p.resolve().parent == parent.resolve() and not p.is_symlink(), 'Invalid result path')
    return p

def verify_binding(binding):
    require(binding['entry_sha256'] == sha(ENTRY), 'Unified entry version changed; author revalidation required')
    for name, expected in binding['files'].items():
        p = Path(name)
        require(p.is_absolute() and inside(p, ROOT) and p.is_file(), 'Invalid bound source: ' + name)
        require(sha(p) == expected, 'Bound source changed: ' + name)

def validate_result(result, dataset, mode):
    require(result.get('schema') == 'digit-unified-results-v2', 'Wrong summary schema')
    require(result.get('dataset') == dataset and result.get('mode') == mode, 'Wrong dataset/mode')
    require(result.get('passed') is True and result.get('complete') is True, 'Incomplete native result')
    require(result.get('evidence_scope') == 'native', 'Native evidence required')
    require(result.get('accuracy_evaluated') == (dataset == 'PA' and mode == 'full20'), 'Wrong accuracy scope')
    pairs = result['paired_rounds']
    n = 5 if mode == 'short5' else 1
    require(len(pairs) == n and [p['round'] for p in pairs] == list(range(1, n + 1)), 'Incomplete paired rounds')
    ratios = []
    for pair in pairs:
        a, b = pair['gids_seconds'], pair['digit_seconds']
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in (a, b)), 'Invalid timing')
        require(math.isclose(pair['speedup'], a / b, rel_tol=1e-9), 'Pair ratio mismatch')
        ratios.append(a / b)
    expected = statistics.median(ratios)
    require(math.isclose(result['speedup'], expected, rel_tol=1e-9), 'Incorrect summary speedup')
    aggregation = 'median_of_five_paired_speedups' if mode == 'short5' else 'single_pair_ratio_of_total_training_time'
    require(result['aggregation'] == aggregation, 'Incorrect aggregation')
    for field in ('cpu_feature_bytes', 'gpu_feature_bytes'):
        require(result['arms']['gids'][field] == result['arms']['digit'][field] > 0, 'Unequal cache capacity')
    epochs = result['epochs']
    expected_keys = {(arm, r, e) for arm in ('gids', 'digit') for r in range(1, n + 1)
                     for e in range(1, 21 if mode == 'full20' else 2)}
    require(len(epochs) == len(expected_keys) and {(e['arm'], e['round'], e['epoch']) for e in epochs} == expected_keys,
            'Missing or duplicate epochs')
    if mode != 'full20':
        require(all(e['measured_batches'] == 300 for e in epochs), 'Incorrect measured window')
    return expected

def view(dataset, mode, state=None):
    state = service(dataset, mode) if state is None else state
    latest = OUT / dataset / mode / 'latest.json'
    if not latest.exists():
        return dict(state='STARTING' if active(state) else 'NOT_STARTED', service=state)
    record = read(latest)
    folder = output(dataset, mode, record['output'])
    invocation = state.get('InvocationID', '')
    if invocation and invocation != record['invocation_id']:
        return dict(state='STARTING' if active(state) else 'STALE', service=state,
                    error='Latest result belongs to another service invocation')
    v = dict(state='RUNNING' if active(state) else 'INCOMPLETE', output=str(folder), service=state)
    if (folder / 'progress.json').exists():
        v['progress'] = read(folder / 'progress.json')
    if active(state):
        return v
    if state.get('Result') != 'success' or state.get('ExecMainStatus') != '0':
        v['state'] = 'FAILED'
    elif (folder / 'completion.json').exists():
        final = read(folder / 'completion.json')
        require(final['invocation_id'] == record['invocation_id'], 'Wrong completion invocation')
        require(final['returncode'] == 0 and final['passed'], 'Failed completion')
        require(sha(folder / 'summary.json') == final['summary_sha256'], 'Published result changed')
        summary = read(folder / 'summary.json')
        validate_result(summary, dataset, mode)
        v.update(state='PASS', summary=summary)
    if (folder / 'error.json').exists():
        v.update(state='FAILED', error=read(folder / 'error.json')['error'])
    return v
