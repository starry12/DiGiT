"""Full independent epoch, same BFS root batches, no feature reads or updates."""
import argparse
import numpy as np
from .common import *

def run():
    heavy_gate();prepared();setup()
    from .dataset import graph_sampler,roots
    from candidates.cache_epoch_v1.core import profile,select_hot,digest_array
    from candidates.uks_group_first_cpu2_v1.affinity import apply_affinity
    import torch
    apply_affinity('digit_full')
    # No native feature store exists in this process.
    from ae.common import host
    require(host()>=160*2**30 and torch.cuda.mem_get_info()[0]>=30*2**30,'Profile resource admission failed')
    graph,arrays,artifact,sampler=graph_sampler('digit',23)
    canonical=np.load(DATA/'canonical.npy');p=cfg()
    def factory(seed):
        require(seed==23,'Independent profile seed required')
        def sample(batch,index):
            return sampler.sample_blocks(graph,torch.from_numpy(batch.copy()).cuda())[0].cpu().numpy()
        return sample
    def event(**kw):write(OUT/'profile_progress.json',kw)
    counts,receipt=profile(canonical,p['graph']['nodes'],factory,event)
    require(receipt['batches']==1179 and receipt['roots']==p['train_count'] and receipt['roots_sha256']==digest_array(roots()),'Profile/root mismatch')
    revpr=np.load(DATA/'revpr_hot.npy');scores=np.memmap(DATA/'revpr.f64',mode='r',dtype=np.float64)
    hot,selection=select_hot(counts,p['cpu_cache_rows'],revpr,scores[revpr])
    target=DATA/'frequency_hot.npy';require(not target.exists(),'Preserve existing profile')
    np.save(target,hot,allow_pickle=False)
    write(DATA/'profile.json',dict(passed=True,source_sha256=verify(),profile=receipt,selection=selection,hot=bind(target),feature_reads=0,optimizer_updates=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock-fds',required=True);a=parser.parse_args()
    validate_inherited_locks(a.lock_fds);run()
