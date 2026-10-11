"""Frozen preparation sources and separate accepted v1/v2 overlays."""
from candidates.uks_group_reuse_v1 import common as v1
from candidates.uks_group_reuse_v2 import common as v2
from candidates.uks_group_reuse_v2.common import (
    ROOT, DATA, PY, DEFAULT_OUTPUT, require, read, write, sha, exclusive,
)

HERE=ROOT/'candidates/uks_group_compare_v2'
OUT=ROOT/'results/uks_group_compare_20261010_v2'
VARIANTS=('gids','v1','v2')
ROUNDS=5

def verify():
    m=read(HERE/'manifest.json')
    require(v1.verify()==m['v1_source'] and v2.verify()==m['v2_source'],'Overlay source changed')
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Candidate changed: '+name)
    for name,h in m['dependencies'].items():require(sha(ROOT/name)==h,'Dependency changed: '+name)
    return sha(HERE/'manifest.json')

def check_prepared(folder=None):
    require(folder is None or folder==DEFAULT_OUTPUT,'Unexpected v2 output path')
    return dict(v1=v1.check_prepared(v1.DEFAULT_OUTPUT),v2=v2.check_prepared(v2.DEFAULT_OUTPUT))

def prepared_hashes():
    return {name:sha(module.DEFAULT_OUTPUT/'prepared.json') for name,module in (('v1',v1),('v2',v2))}

def matched_roots(arm):
    require(arm in ('gids','digit'),'Unknown arm')
    from candidates.uks_freq_bfs_v1.common import load_roots
    return load_roots('digit')

def stages():
    result=[('gpu_check',None,None,None)]
    result += [('smoke_'+v,'smoke',v,None) for v in VARIANTS]
    result += [('round%d_%s'%(i,v),'performance',v,i) for i in range(1,ROUNDS+1) for v in VARIANTS]
    return result
