"""CPU-only order loading; native BFS imports are confined to a preparer process."""
import hashlib
import numpy as np
from .common import *
from .mixing import mix_bfs_chunks


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def validate_roots(roots,train):
    require(roots.dtype==np.int64 and roots.ndim==1 and
            len(roots)==len(train) and np.array_equal(np.sort(roots),np.sort(train)),
            'Order loses, duplicates or substitutes training roots')


def extract(edges,train,n,chunk=1000000):
    require(edges.ndim==2 and 2 in edges.shape,'Invalid original edge shape')
    if edges.shape[0]!=2:edges=edges.T
    mask=np.zeros(n,dtype=bool);mask[train]=True;parts=[]
    for lo in range(0,edges.shape[1],chunk):
        e=edges[:,lo:lo+chunk]
        require(e.size==0 or (e.min()>=0 and e.max()<n),'Invalid source endpoint')
        keep=mask[e[0]] & mask[e[1]] & (e[0]!=e[1])
        if keep.any():parts.append(np.array(e[:,keep],copy=True))
    return np.concatenate(parts,axis=1) if parts else np.empty((2,0),dtype=np.int64)


def load_roots(binding,arm,epoch,train):
    item=binding['orders'][arm][str(epoch)]
    require(sha(item['path'])==item['sha256'],'Root order file changed')
    roots=np.load(item['path'],allow_pickle=False)
    validate_roots(roots,train)
    require(digest(roots)==item['digest'],'Root sequence digest differs')
    require(item['shuffle']['policy']==cfg()['arms'][arm]['order_policy'],'Wrong ordering policy binding')
    return roots


def prepare_orders(folder):
    # This subprocess exits before any measured worker starts. Canonical imports
    # avoid the historical double-load of DiGiTSamplerCUDA under two names.
    setup()
    from digit.bfs_order import build_bfs_train_order,shuffle_bfs_batch_blocks
    from ae.pa_sage.common import splits
    from .worker import identity,check_inputs
    import torch
    binding=read(folder/'inputs.json');check_inputs(binding);p=cfg()
    require(not torch.cuda.is_initialized(),'BFS preparation initialized CUDA')
    before=time.perf_counter();train=splits('train')
    desc=source_config()['source_contract']['original_edges']
    edges=np.load(desc['path'],mmap_mode='r')
    induced=extract(edges,train,p['graph']['nodes'])
    canonical,statistics=build_bfs_train_order(induced,train,p['graph']['nodes'])
    validate_roots(canonical,train)
    canonical_seconds=time.perf_counter()-before
    dest=folder/'orders';dest.mkdir();orders={a:{} for a in p['ablation_order']}
    np.save(dest/'canonical.npy',canonical,allow_pickle=False)
    generated=[dest/'canonical.npy'];shuffle_seconds=0.0;random_seconds=0.0
    per_arm_seconds={a:0.0 for a in p['ablation_order']}
    expected=read(ROOT/p['base_layout']/'orders.json')['hashes']
    for epoch in range(max(p['epochs'], p['smoke_epochs'])):
        for arm in p['ablation_order']:
            started=time.perf_counter()
            policy=p['arms'][arm]['order_policy']
            if policy=='bfs_mixed_chunks':
                roots,meta=mix_bfs_chunks(canonical,p['batch_size'],p['arms'][arm]['chunk_size'],p['seed'],epoch)
            elif policy=='bfs_batch_blocks':
                roots,meta=shuffle_bfs_batch_blocks(canonical,p['batch_size'],p['seed'],epoch)
                meta['policy']='bfs_batch_blocks'
            else:
                require(policy=='random','Unknown order policy')
                roots=np.random.default_rng(np.random.SeedSequence([p['seed'],epoch])).permutation(train)
                meta=dict(policy='random')
                require(digest(roots)==expected['s0_e%d'%epoch],'Original AE root order differs')
            elapsed=time.perf_counter()-started
            per_arm_seconds[arm]+=elapsed
            if policy=='random':random_seconds+=elapsed
            else:shuffle_seconds+=elapsed
            validate_roots(roots,train)
            path=dest/('%s_e%d.npy'%(arm,epoch));np.save(path,roots,allow_pickle=False)
            generated.append(path)
            orders[arm][str(epoch)]=dict(path=str(path),sha256=sha(path),digest=digest(roots),
                                       examples=len(roots),shuffle=meta)
    receipt=dict(passed=True,canonical_seconds=canonical_seconds,bfs_shuffle_seconds=shuffle_seconds,
        bfs_preparation_seconds=canonical_seconds+shuffle_seconds,random_preparation_seconds=random_seconds,
        total_preparation_seconds=time.perf_counter()-before,statistics=statistics,
        policy='DiGiT order control: random, fixed BFS batches, epoch-rotated mixed BFS chunks',
        per_arm_order_seconds=per_arm_seconds,mixing=p['bfs_mixing'],
        graph_relation='Undirected source edges have the same connectivity as the AE bidirectional graph; self loops ignored',
        hot_set='unchanged original AE full_cpu_rows.npy; no new profiling pass',
        seed=p['seed'],epochs_prepared=list(range(max(p['epochs'],p['smoke_epochs']))),orders=orders,cuda_initialized=torch.cuda.is_initialized(),
        source_edges_sha256=desc['sha256'],train_split=source_config()['splits']['train'])
    require(not receipt['cuda_initialized'],'BFS preparation initialized CUDA')
    check_inputs(binding)
    for path in generated:
        binding['files'][str(path)]=dict(sha256=sha(path),identity=identity(path))
    binding['orders']=orders
    write(folder/'bfs_preparation.json',receipt)
    path=folder/'bfs_preparation.json'
    binding['files'][str(path)]=dict(sha256=sha(path),identity=identity(path))
    binding['bfs_preparation']=receipt
    write(folder/'inputs.json',binding)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    prepare_orders(parser.parse_args().output.resolve())
