"""Immutable experiment binding; retain the accepted preparation unchanged."""
from candidates.uks_group_reuse_v1.common import (
    ROOT, DATA, PY, DEFAULT_OUTPUT, require, read, write, sha, exclusive,
    check_prepared, verify as verify_parent,
)

HERE = ROOT/'candidates/uks_matched_bfs_v1'
OUT = ROOT/'results/uks_matched_bfs_20261010_v1'
VARIANTS = ('gids', 'old', 'reuse')
ROUNDS = 5


def verify():
    manifest = read(HERE/'manifest.json')
    require(verify_parent() == manifest['parent_sha256'], 'Parent changed')
    for name, expected in manifest['files'].items():
        require(sha(HERE/name) == expected, 'Source changed: '+name)
    for name, expected in manifest['dependencies'].items():
        require(sha(ROOT/name) == expected, 'Dependency changed: '+name)
    return sha(HERE/'manifest.json')


def matched_roots(arm):
    require(arm in ('gids', 'digit'), 'Unknown arm')
    from candidates.uks_freq_bfs_v1.common import load_roots
    # Always load the accepted BFS window, including its original warmup prefix.
    return load_roots('digit')


def stages():
    result = [('gpu_check', None, None, None)]
    result += [('smoke_'+v, 'smoke', v, None) for v in VARIANTS]
    for number in range(1, ROUNDS+1):
        result += [('round%d_%s' % (number, v), 'performance', v, number)
                   for v in VARIANTS]
    return result
