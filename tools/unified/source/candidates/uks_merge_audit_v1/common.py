from candidates.uks_mixed_1k2k_v1.common import *
from candidates.uks_mixed_1k2k_v1 import common as parent
HERE=ROOT/'candidates/uks_merge_audit_v1'
OUT=ROOT/'results/uks_merge_audit_20260930_v1'
def verify():
    m=read(HERE/'manifest.json');require(parent.verify()==m['parent_sha256'],'Parent changed')
    for k,v in m['files'].items():require(sha(HERE/k)==v,'Audit changed: '+k)
    return sha(HERE/'manifest.json')
def check_ready():
    verify();parent.check_ready()
