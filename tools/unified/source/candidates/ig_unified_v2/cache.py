
import hashlib,json,random
from pathlib import Path
import numpy as np
from candidates.unified_training_v2.order import epoch_batches,roots_digest
from candidates.group_cache_migration_v1.core import profile,select_hot,digest_array

def accepted_hot(directory,config):
    d=Path(directory);r=json.loads((d/'profile.json').read_text())
    from .common import verify
    if not r['complete'] or r['source_sha256']!=verify() or r.get('policy')!=config['cache_profile']:
        raise ValueError('Missing current cache profile')
    from candidates.unified_training_v2.profiling import validate_profile
    validate_profile(r,config,r['canonical_root_count'])
    hot=np.load(d/'cache_hot.npy',allow_pickle=False)
    if digest_array(hot)!=r['selection']['hot_sha256']:raise ValueError('Changed hot payload')
    return hot,r,None

def prepare(graph,sampler,canonical,nodes,revpr,output,check,event,config):
    import torch,dgl
    from .common import verify
    output=Path(output);target=output/'profile.json'
    if target.exists():
        hot,r,_=accepted_hot(output,config)
        if r['canonical_sha256']!=digest_array(canonical):raise ValueError('Profile roots changed')
        sampler.random_seed=0;sampler._cuda_call_counter=0
        return hot,r
    def factory(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);dgl.seed(seed)
        sampler.random_seed=seed;sampler._cuda_call_counter=0
        def sample(roots,index):
            inputs,outputs,blocks=sampler.sample_blocks(graph,torch.from_numpy(np.array(roots,copy=True)).cuda())
            if not torch.equal(outputs,torch.from_numpy(np.array(roots,copy=True)).to(outputs.device)):
                raise ValueError('Profile changed root order')
            return inputs.cpu().numpy().copy()
        return sample
    from candidates.unified_training_v2.profiling import window_profile,short_hot,validate_profile
    if config['cache_profile']['scope']=='short_window':
        counts,r=window_profile(canonical,nodes,factory,config,event,check)
        hot,selection=short_hot(counts,revpr,check)
    else:
        counts,r=profile(canonical,nodes,factory,event,check,batch_iterator=epoch_batches(canonical,0,block_nodes=1024))
        expected=roots_digest(epoch_batches(canonical,0,block_nodes=1024))
        if r['roots_sha256']!=expected:raise ValueError('Profile roots differ from fixed cache-selection order')
        if r['observed_nodes']<len(revpr):raise ValueError('Insufficient observed IG nodes')
        hot,selection=select_hot(counts,len(revpr),revpr,check=check)
    r.update(policy=dict(config['cache_profile']),canonical_root_count=len(canonical))
    validate_profile(r,config,len(canonical))
    with (output/'cache_hot.npy').open('xb') as f:np.save(f,hot,allow_pickle=False)
    r.update(source_sha256=verify(),selection=selection,order_policy='BFS256_epoch0_prefix' if config['cache_profile']['scope']=='short_window' else 'bfs_block_shuffle',
             feature_cache_created=False,formal_sampler_seed=0,formal_sampler_call_counter=0)
    with target.open('x') as f:json.dump(r,f,indent=2)
    sampler.random_seed=0;sampler._cuda_call_counter=0
    torch.cuda.synchronize();return hot,r
