from candidates.uks_1kb_pair_v1.common import *
from candidates.uks_1kb_pair_v1 import common as parent
HERE=ROOT/'candidates/uks_1kb_retry_v1'
OUT=ROOT/'results/uks_1kb_pair_20260930_v1/retry1'
def verify():
    m=read(HERE/'manifest.json');require(parent.verify()==m['parent_sha256'],'Parent changed')
    for k,v in m['files'].items():require(sha(HERE/k)==v,'Retry source changed: '+k)
    for k,v in m['evidence'].items():require(sha(ROOT/k)==v,'Accepted short evidence changed: '+k)
    return sha(HERE/'manifest.json')
def check_ready():
    verify();parent.check_ready()
