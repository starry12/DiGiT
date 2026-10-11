from candidates.uks_mixed_1k2k_v1.common import *
from candidates.uks_mixed_1k2k_v1 import common as parent
HERE=ROOT/'candidates/uks_freq_bfs_v1'
OUT=ROOT/'results/uks_freq_bfs_20260930_v1'
LARGE=Path('/mnt/n0/digit/uks_freq_bfs_20260930_v1')
FREQ=Path('/mnt/n0/digit/uks_native_20260929_v1/freq_hot.npy')
def verify():
    m=read(HERE/'manifest.json');require(parent.verify()==m['parent_sha256'],'Parent changed')
    for k,v in m['files'].items():require(sha(HERE/k)==v,'Source changed: '+k)
    for k,v in m['evidence'].items():require(sha(ROOT/k)==v,'Evidence changed: '+k)
    return sha(HERE/'manifest.json')
def check_ready():
    verify();parent.check_ready()
    r=read(native_parent.OUT/'profile/report.json')
    require(r['passed'] and r['batches']==100 and r['seed']==23 and sha(FREQ)==r['hot_sha256'],'Freq profile changed')
def hot_path(arm):return DATA/'rank/hot_nodes.npy' if arm=='gids' else FREQ
def load_roots(arm):
    import numpy as np
    if arm=='gids':return np.load(DATA/'synthetic/roots.npy',mmap_mode='r')[:320*1024].copy()
    r=read(OUT/'bfs_receipt.json');require(r['passed'] and r['source_sha256']==verify() and sha(LARGE/'digit_roots.npy')==r['window_sha256'],'BFS order changed')
    return np.load(LARGE/'digit_roots.npy')
