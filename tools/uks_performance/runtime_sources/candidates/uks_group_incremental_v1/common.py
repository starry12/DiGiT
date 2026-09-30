from candidates.uks_sampling_profile_v1.common import *
from candidates.uks_sampling_profile_v1 import common as parent
HERE=ROOT/'candidates/uks_group_incremental_v1'
OUT=ROOT/'results/uks_group_incremental_20260930_v1'
BINARY=HERE/'runtime/UKSGroupIncrementalCUDA.so'
def verify():
    m=read(HERE/'manifest.json');require(parent.verify()==m['parent_sha256'],'Parent changed')
    for k,v in m['files'].items():require(sha(HERE/k)==v,'Source changed: '+k)
    for k,v in m['dependencies'].items():require(sha(ROOT/k)==v,'Dependency changed: '+k)
    return sha(HERE/'manifest.json')
def check_ready():
    verify();parent.check_ready();r=read(HERE/'runtime/build_receipt.json')
    require(r['source_sha256']==verify() and sha(BINARY)==r['binary_sha256'],'Build changed')
def raw_extension():
    import importlib.util,sys
    name='UKSGroupIncrementalCUDA'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,str(BINARY));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]
def extension():
    import os,types
    m=raw_extension();p=types.SimpleNamespace(**{k:getattr(m,k) for k in dir(m) if not k.startswith('__')})
    kernel=os.environ.get('UKS_GROUP_KERNEL','legacy');require(kernel in ('legacy','incremental'),'Invalid kernel')
    if kernel=='incremental':p.sample_group_aware_i32_uva64=m.sample_group_aware_i32_uva64_incremental
    return p
