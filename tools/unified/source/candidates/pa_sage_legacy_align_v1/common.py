import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/pa_sage_legacy_align_20260926_v1'
PY = '/home/embed/miniconda3/envs/gids/bin/python'
MODULE = 'candidates.pa_sage_legacy_align_v1'
PARENT = ROOT / 'candidates/io_accounting_v1'
BINARY = HERE / 'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
GPU_UUID = 'GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def identity(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def progress(folder, stage, **extra):
    v = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra)
    write(Path(folder) / 'progress.json', v)
    print(json.dumps(v), flush=True)


def setup():
    os.environ.update(DGLBACKEND='pytorch', PYTHONDONTWRITEBYTECODE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path[:0] = [str(HERE / 'runtime'), str(ROOT / 'evaluation')]


def native_sources():
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (HERE / 'native').rglob('*') if p.is_file()}
    for rel, digest in read(PARENT / 'manifest.json')['files'].items():
        if rel.startswith('native/bam/include/'):
            p = PARENT / rel
            require(sha(p) == digest, 'Frozen parent header changed: ' + rel)
            sources[str(p.relative_to(ROOT))] = digest
    sources['bam/build/lib/libnvm.so'] = sha(ROOT / 'bam/build/lib/libnvm.so')
    return sources


def verify():
    m = read(HERE / 'manifest.json')
    for section in ('files', 'dependencies'):
        for path, digest in m[section].items():
            require(sha(ROOT / path) == digest, 'Source changed: ' + path)
    require(sha(BINARY) == m['binary_sha256'], 'Native binary changed')
    require(sha(OUT / 'inputs.json') == m['inputs_sha256'], 'Input binding changed')
    return sha(HERE / 'manifest.json')


def check_inputs(binding):
    for path, v in binding['files'].items():
        require(identity(path) == v['identity'], 'Input file changed: ' + path)


def validate_completion(report):
    p = read(HERE / 'protocol.json')
    expected = p['smoke_examples'] if report['smoke'] else p['train_count']
    updates = (expected + p['batch_size'] - 1) // p['batch_size']
    require(report['passed'] and report['examples'] == expected and report['updates'] == updates,
            'Incomplete epoch')
    pipe = report['pipeline']
    require(pipe['drained'] and pipe['sampled_batches'] == pipe['read_batches'] ==
            pipe['delivered_batches'] == updates, 'Window tail was not fully trained')
    require(report['fetch_count'] == report['sample_count'] == updates,
            'Sampling/fetch counters do not cover every batch')
    require(report['root_coverage_exact'] and report['tail_size'] == expected - (updates - 1) * p['batch_size'],
            'Incomplete root coverage or tail')
    require(report['logical_feature_requests'] == sum(report['feature_access'].values()),
            'Feature rows not reconciled')
    require(report['gpu_cache']['requests'] == report['feature_access']['gpu_ssd'],
            'GPU feature requests not reconciled')
    io = report['device_io']
    require(io['outstanding'] == 0 and io['submitted_commands'] == io['completed_commands'],
            'Incomplete native I/O')
    require(io['completed_bytes'] > 0 and io['completed_bytes'] % 512 == 0,
            'Expected real sector-aligned SSD reads')
    require(math.isfinite(report['timing']['e2e_seconds']) and
            report['timing']['e2e_seconds'] > 0 and report['finite_parameters'], 'Invalid training')
    if report['smoke']:
        require(report['smoke_checks']['features_bit_exact'] and report['smoke_checks']['sampled_edges_exact'],
                'Native smoke did not validate features and graph')
    return True
