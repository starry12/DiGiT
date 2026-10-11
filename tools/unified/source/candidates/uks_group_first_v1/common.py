"""Source and binary binding for the isolated sampler."""
from candidates.uks_group_reuse_v2.common import ROOT,read,write,sha,require,PY
HERE=ROOT/'candidates/uks_group_first_v1'
OUT=ROOT/'results/uks_group_first_20261010_v1'
BINARY=HERE/'runtime/UKSGroupFirstCUDA.so'
from candidates.uks_group_compare_v2.common import (
    VARIANTS,ROUNDS,DEFAULT_OUTPUT,check_prepared,prepared_hashes,exclusive,matched_roots,
)

def verify():
    from candidates.uks_group_compare_v2.common import verify as verify_comparison
    verify_comparison()
    m=read(HERE/'manifest.json')
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Candidate changed: '+name)
    for name,h in m['dependencies'].items():require(sha(ROOT/name)==h,'Dependency changed: '+name)
    return sha(HERE/'manifest.json')

def extension():
    import importlib.util,sys
    r=read(HERE/'runtime/build_receipt.json')
    require(r['source_sha256']==verify() and r['binary_sha256']==sha(BINARY),'Unverified sampler build')
    name='UKSGroupFirstCUDA'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,str(BINARY))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);sys.modules[name]=m
    m=sys.modules[name]
    require(m.SELECTION_POLICY=='groups_first_without_replacement','Wrong sampler policy')
    return m
