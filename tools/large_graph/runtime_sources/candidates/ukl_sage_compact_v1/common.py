"""Reuse frozen isolation helpers; never import a device backend."""
from pathlib import Path
from candidates.uks_sage_epoch_v1.common import (
    ROOT, GRID, require, read, sha, digest, write_new, cpu_gate,
    heavy_gate, heavy_lock, require_grid_complete,
)
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/ukl_sage_compact_20260925_v1'
LARGE = Path('/mnt/n0/digit/ukl_sage_compact_20260925_v1')
MAX_NODES = 4096
MAX_EDGES = 131072


def bounded(nodes, edges, fixture):
    require(type(nodes) is int and 0 < nodes < 2**31 and type(edges) is int and 0 <= edges < 2**63,
            'Invalid node/edge extent')
    if fixture:
        require(nodes <= MAX_NODES and edges <= MAX_EDGES, 'Bounded fixture only')
        cpu_gate()
    else:
        heavy_gate()


def verify():
    m = read(HERE / 'manifest.json')
    for n, h in m['files'].items():require(sha(HERE / n) == h, 'UKL source changed: ' + n)
    for n, h in m['dependencies'].items():require(sha(ROOT / n) == h, 'Dependency changed: ' + n)
    return sha(HERE / 'manifest.json')
