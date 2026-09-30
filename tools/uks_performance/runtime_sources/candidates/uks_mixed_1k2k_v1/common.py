from candidates.uks_revpr_diagnostic_v1.common import *
from candidates.uks_group_incremental_v1 import common as sampler_parent
from candidates.uks_native_v1 import common as native_parent
HERE=ROOT/'candidates/uks_mixed_1k2k_v1'
OUT=ROOT/'results/uks_mixed_1k2k_20260930_v1'
BINARY=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
def verify():
    m=read(HERE/'manifest.json');require(sampler_parent.verify()==m['sampler_parent_sha256'],'Sampler parent changed')
    for k,v in m['files'].items():require(sha(HERE/k)==v,'Source changed: '+k)
    for k,v in m['dependencies'].items():require(sha(ROOT/k)==v,'Dependency changed: '+k)
    return sha(HERE/'manifest.json')
def binary_receipt():
    r=read(HERE/'runtime/build_receipt.json');require(r['source_sha256']==verify() and r['binary_sha256']==sha(BINARY),'Binary changed');return r
def check_ready():
    sampler_parent.check_ready();verify();binary_receipt()
def select_sampler(arm,seed):
    from candidates.uks_native_v1.binding import graph_sampler
    result=graph_sampler(arm,seed)
    if arm=='digit':
        import os,digit.sampler as m
        os.environ['UKS_GROUP_KERNEL']='incremental';m._cuda_extension=sampler_parent.extension()
    return result
