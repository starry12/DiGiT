import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from candidates.pa_sage_cache_policy_v1.common import ROOT, read, sha, require, write_new
from candidates.pa_sage_cache_policy_v2.common import GRID, require_grid_complete

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/pa_sage_cpu_capacity_20260925_v1'
LARGE = Path('/mnt/n0/digit/pa_sage_cpu_capacity_20260925_v1')
PY = '/home/embed/miniconda3/envs/gids/bin/python'
MODULE = 'candidates.pa_sage_cpu_capacity_v1'
GPU = '2'
GPU_UUID = 'GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'
PERCENTAGES = (0, 5, 10, 20)
ARMS = tuple('cpu%02d' % x for x in PERCENTAGES)
PARENT = ROOT / 'candidates/io_accounting_v1'
BINARY = HERE / 'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
SCRATCH_BYTES = 16 * 2**20


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def identity(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify():
    m = read(HERE / 'manifest.json')
    for name, expected in m['files'].items():
        require(sha(HERE / name) == expected, 'Controller source changed: ' + name)
    for name, expected in m['dependencies'].items():
        require(sha(ROOT / name) == expected, 'Controller dependency changed: ' + name)
    return sha(HERE / 'manifest.json')


def native_source_id():
    paths = list((HERE / 'native').rglob('*')) + [HERE / x for x in
        ('backend.py', 'runtime.py', 'counters.py', 'counter_math.py', 'build.py')]
    return object_sha({str(p.relative_to(HERE)): sha(p) for p in sorted(paths) if p.is_file()})


def binary_receipt():
    source = verify()
    receipt = read(HERE / 'runtime/build_receipt.json')
    require(receipt['source_sha256'] == source and receipt['binary_sha256'] == sha(BINARY),
            'Missing or stale isolated CPU capacity binary')
    return receipt


def heavy_gate():
    # Called BEFORE mkdir, nvidia-smi, CUDA imports, full array scans or a build.
    require_grid_complete(read(GRID / 'status.json'))
    require(__debug__ and os.geteuid() == 0, 'Native controller requires local administrator execution without -O')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Expected physical GPU 2')


@contextlib.contextmanager
def lock(path, create=True):
    with Path(path).open('a+' if create else 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield f


def progress(folder, stage, **kwargs):
    record = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **kwargs)
    write(Path(folder) / 'progress.json', record)
    print(json.dumps(record), flush=True)


def complete_handshake(folder, report):
    write(folder / 'report.json', report)
    write(folder / 'worker_ready.json', dict(passed=True, report_sha256=sha(folder / 'report.json')))
    deadline = time.monotonic() + 180
    while not (folder / 'release_worker.json').exists():
        require(time.monotonic() < deadline, 'Monitor/controller release timed out')
        time.sleep(.25)
    require(read(folder / 'release_worker.json').get('passed') is True, 'Controller rejected monitoring or worker report')
