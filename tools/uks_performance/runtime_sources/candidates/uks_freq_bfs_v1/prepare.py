import time,subprocess,numpy as np
from .common import *
def main():
    check_ready()
    from candidates.uks_native_v1.binding import check,protocol
    check();started=time.time();p=protocol()
    if (OUT/'bfs_receipt.json').exists():
        load_roots('digit');print('Reusing accepted BFS preparation');return
    LARGE.mkdir(parents=True,exist_ok=True)
    require(not (LARGE/'canonical.bin').exists(),'Preserve previous unfinished preparation')
    train=np.load(DATA/'synthetic/train.npy',mmap_mode='r');n=p['nodes']
    cmd=[str(HERE/'bin/bfs'),str(DATA/'csc/indptr.npy'),str(DATA/'csc/indices.npy'),str(DATA/'synthetic/train.npy'),str(n),str(len(train)),str(LARGE/'canonical.bin')]
    with (OUT/'bfs_build.log').open('x') as log:
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=log,text=True,check=True)
    stats=__import__('json').loads(r.stdout);canonical=np.fromfile(LARGE/'canonical.bin',dtype=np.int64)
    require(np.array_equal(np.sort(canonical),train),'BFS does not cover the exact training set')
    np.save(LARGE/'canonical.npy',canonical)
    from candidates.pa_sage_bidir_native_v2.runtime.digit.bfs_order import shuffle_bfs_batch_blocks
    roots,meta=shuffle_bfs_batch_blocks(canonical,1024,0,0);window=roots[:320*1024].copy();np.save(LARGE/'digit_roots.npy',window)
    old=load_roots('gids');overlap=int(np.intersect1d(window,old).size)
    write(OUT/'bfs_receipt.json',dict(passed=True,source_sha256=verify(),window_sha256=sha(LARGE/'digit_roots.npy'),canonical_sha256=sha(LARGE/'canonical.npy'),canonical_train_nodes=len(train),stats=stats,shuffle=meta,window_nodes=len(window),old_gids_window_overlap=overlap,preparation_seconds=time.time()-started,full_epoch=False,raw_ssd_writes=False))
if __name__=='__main__':main()
