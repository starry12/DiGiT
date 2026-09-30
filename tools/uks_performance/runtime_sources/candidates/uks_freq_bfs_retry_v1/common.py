from candidates.uks_freq_bfs_v1.common import *
from candidates.uks_freq_bfs_v1 import common as parent
HERE=ROOT/'candidates/uks_freq_bfs_retry_v1'
OUT=ROOT/'results/uks_freq_bfs_20260930_v1/retry1'
def verify():
    m=read(HERE/'manifest.json');require(parent.verify()==m['parent_sha256'],'Parent changed')
    for k,v in m['files'].items():require(sha(HERE/k)==v,'Retry source changed: '+k)
    for k,v in m['evidence'].items():require(sha(ROOT/k)==v,'Prior evidence changed: '+k)
    return sha(HERE/'manifest.json')
def check_ready():verify();parent.check_ready();load_roots('digit')
def coverage(region):
    return all(region['serving'][k]>0 for k in ('cpu_served_rows','gpu_hit_rows','ssd_served_rows'))
