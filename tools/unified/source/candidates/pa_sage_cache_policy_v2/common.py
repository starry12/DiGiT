"""Small-file provenance and a fail-closed gate for deferred expensive work."""
import json
from pathlib import Path
from candidates.pa_sage_cache_policy_v1.common import ROOT, sha, require, read, write_new

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/pa_sage_cache_policy_20260925_v2'
GRID = ROOT / 'results/pa_sage_layout_continue_20260925_v4'
PARENT = ROOT / 'candidates/io_accounting_v1'
BINARY = HERE / 'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
SCRATCH_BYTES = 16 * 2**20


def verify():
    m = read(HERE / 'manifest.json')
    for name, expected in m['files'].items():
        require(sha(HERE / name) == expected, 'Changed v2 source: ' + name)
    for name, expected in m['dependencies'].items():
        require(sha(ROOT / name) == expected, 'Changed dependency: ' + name)
    return sha(HERE / 'manifest.json')


def require_grid_complete(state):
    require(state.get('complete') is True and state.get('passed') is True,
            'Deferred until the existing layout grid completes successfully')
    require(len(state.get('completed', [])) == 15 and len(set(state['completed'])) == 15,
            'Existing grid is missing completed points')


def protected_now():
    before = read(OUT / 'protected_before.json')
    return {name: sha(ROOT / name) for name in before}


def binary_receipt():
    manifest = verify()
    receipt = read(HERE / 'runtime/build_receipt.json')
    require(receipt['source_sha256'] == manifest and receipt['binary_sha256'] == sha(BINARY),
            'Missing or stale isolated native build')
    return receipt
