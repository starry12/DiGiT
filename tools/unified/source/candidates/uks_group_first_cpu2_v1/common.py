from candidates.uks_group_compare_v2.common import (
    ROOT,DATA,PY,DEFAULT_OUTPUT,read,write,sha,require,exclusive,
    check_prepared,prepared_hashes,matched_roots,
)
from candidates.uks_group_first_compare_v1.common import verify as verify_parent
from candidates.uks_group_compare_v2.common import verify as verify_weighted
from candidates.uks_group_first_v1.common import verify as verify_first,BINARY
HERE=ROOT/'candidates/uks_group_first_cpu2_v1'
OUT=ROOT/'results/uks_group_first_cpu2_20261010_v1'
VARIANTS=('gids','weighted','first')
ROUNDS=5
POLICIES={'gids':'gids_standard','weighted':'size_weighted_without_replacement',
          'first':'groups_first_without_replacement'}

def verify():
    m=read(HERE/'manifest.json')
    require(verify_parent()==m['parent_source'],'Parent source changed')
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Source changed: '+name)
    for name,h in m['dependencies'].items():require(sha(ROOT/name)==h,'Dependency changed: '+name)
    return sha(HERE/'manifest.json')

def stages():
    result=[('gpu_check',None,None,None)]
    result += [('smoke_'+v,'smoke',v,None) for v in VARIANTS]
    result += [('round%d_%s'%(i,v),'performance',v,i) for i in range(1,ROUNDS+1) for v in VARIANTS]
    return result
