"""A new protocol; historical candidates and deployed AE remain immutable."""
import json
import os
import sys
import time
from pathlib import Path
from ae.common import sha, write, require
from ae.pa_sage.common import source_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'candidates/pa_sage_bidir_native_v2'
EXACT = ROOT / 'candidates/pa_sage_ablation_cache_v3'
P = HERE / 'protocol.json'
OUT = ROOT / 'results/pa_sage_revpr_node_20261009_v1'
PY = '/home/embed/miniconda3/envs/gids/bin/python'
MODULE = 'candidates.pa_sage_revpr_node_v1'
GPU = '2'
GPU_UUID = 'GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'


def read(path):
    return json.loads(Path(path).read_text())


def cfg():
    p = read(P)
    require(p['epochs'] == 1 and p['evaluation'] == 'disabled', 'Performance pilot only')
    original = read(PARENT / 'protocol.json')
    for key in ('seed', 'fanouts', 'batch_size', 'optimizer', 'hidden', 'classes',
                'layers', 'dropout', 'graph', 'gpu_cache_bytes', 'cpu_cache_rows',
                'base_layout', 'data', 'metadata_mode', 'bfs_enabled'):
        require(p[key] == original[key], 'Unrelated AE setting changed: ' + key)
    require(p['page_bytes'] == 4096 and not p['window_buffer'] and not p['accumulator'],
            'The 512-byte execution path must not be enabled')
    return p


def backend_binary(arm):
    base = EXACT if cfg()['arms'][arm]['backend'] == 'exact' else ROOT / 'candidates/io_accounting_v1'
    return base / 'runtime/BAM_Feature_Store/BAM_Feature_Store.so'


def setup(arm=None):
    from candidates.pa_sage_bidir_native_v2.common import setup as parent_setup
    parent_setup()
    if arm and cfg()['arms'][arm]['backend'] == 'exact':
        sys.path.insert(0, str(EXACT / 'runtime'))


def progress(output, stage, **kw):
    value = dict(stage=stage, pid=os.getpid(), time_unix=time.time(), **kw)
    write(Path(output) / 'progress.json', value)
    print(json.dumps(value), flush=True)


def verify():
    m = read(HERE / 'manifest.json')
    for rel, digest in m['files'].items():
        require(sha(HERE / rel) == digest, 'Pilot source changed: ' + rel)
    for rel, digest in m['dependencies'].items():
        require(sha(ROOT / rel) == digest, 'Pilot dependency changed: ' + rel)
    from candidates.pa_sage_ablation_cache_v3.common import verify as exact_verify
    from candidates.pa_sage_bidir_native_v2.common import verify as parent_verify
    require(exact_verify() == m['exact_candidate_sha256'], 'Exact-row backend changed')
    require(parent_verify() == m['parent_candidate_sha256'], 'Original AE runtime changed')
    cfg()
    return sha(HERE / 'manifest.json')
