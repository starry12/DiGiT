"""Complete frozen IG root sequence, independent profiling seed, fixed layout."""
import copy
import numpy as np
from candidates.group_cache_migration_v1.common import ROOT, read, write, require, sha, verify
from candidates.group_cache_migration_v1.prepare import check_prepared
from candidates.group_cache_migration_v1.core import profile, select_hot, digest_array, validate_order

def overlay(arrays,manifest):
    receipt=check_prepared('IG');arrays=dict(arrays);manifest=copy.deepcopy(manifest)
    for key,name in [('reordered_indptr','ptr.i64'),('reordered_indices','units.i64')]:
        v=receipt['files'][name];a=np.memmap(v['path'],mode='r',dtype='<i8')
        arrays[key]=a;manifest['files'][key]=dict(v,shape=list(a.shape),dtype='<i8',format='raw')
    manifest.setdefault('metadata',{})['group_reuse']=receipt['validation'];return arrays,manifest

def prepare_hot(graph,sampler,output,mark,smoke):
    import torch
    p=read(ROOT/'candidates/group_cache_migration_v1/ig/protocol.json');source=verify()
    base=output if smoke else output.parent.parent/'smoke_digit_full/digit_full'
    target=base/'full_epoch_hot.npy';receipt_path=base/'full_epoch_profile.json'
    if not smoke:
        r=read(receipt_path);require(r['complete'] and r['source_sha256']==source and sha(target)==r['hot_file_sha256'],'Missing accepted full IG profile')
        hot=np.load(target,allow_pickle=False);require(digest_array(hot)==r['selection']['hot_sha256'],'Changed hot IDs')
        sampler.random_seed=0;sampler._cuda_call_counter=0
        return hot,r
    order=np.load(ROOT/p['source_train_order'],mmap_mode='r');ready=read(ROOT/p['source_order_ready'])
    require(order.shape==(p['train_nodes'],) and sha(ROOT/p['source_train_order'])==ready['files']['train_order.npy'],'Full IG root source changed')
    validate_order(order,np.arange(p['train_nodes'],dtype=np.int64),p['nodes'])
    frozen=np.load(ROOT/p['train_order'],mmap_mode='r');require(np.array_equal(frozen,order[:len(frozen)]),'IG window/profile roots not aligned')
    def factory(seed):
        import random,dgl
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);dgl.seed(seed)
        sampler.random_seed=seed;sampler._cuda_call_counter=0
        def sample(roots,index):
            inp,out,blocks=sampler.sample_blocks(graph,torch.from_numpy(np.array(roots,copy=True)).cuda())
            ids=inp.cpu().numpy().copy();del inp,out,blocks;return ids
        return sample
    def event(stage,**kw):mark(stage,**kw)
    sequence=(order[i:i+1024] for i in range(0,len(order),1024))
    counts,r=profile(order,p['nodes'],factory,event,batch_iterator=sequence)
    revpr=np.load(ROOT/'data/igb/gids_cpu_rows.npy',mmap_mode='r')
    # Every training root is counted once, so observed >= train_nodes > cache rows.
    require(r['observed_nodes']>=p['cpu_cache_rows'],'Unexpected insufficient positive observations')
    hot,selection=select_hot(counts,p['cpu_cache_rows'],revpr);del counts
    with target.open('xb') as f:np.save(f,hot,allow_pickle=False)
    r.update(source_sha256=source,selection=selection,hot_file_sha256=sha(target),
             order_policy='complete original IG random order; same prefix as both formal arms',
             gpu_feature_cache_created=False,formal_sampler_seed=0,formal_sampler_call_counter=0)
    write(receipt_path,r);sampler.random_seed=0;sampler._cuda_call_counter=0
    torch.cuda.synchronize();torch.cuda.empty_cache();return hot,r
