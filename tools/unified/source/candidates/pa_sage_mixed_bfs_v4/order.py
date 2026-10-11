"""Use the accepted BFS chunk algorithm and fixed, outcome-independent LR rule."""
import numpy as np
from candidates.pa_sage_bfs_decay_v1.mixing import mix_bfs_chunks,digest
from .common import *


def learning_rate(p,epoch):
    require(type(epoch) is int and 0<=epoch<p['epochs'],'Bad epoch')
    rule=p['lr_schedule']
    return p['optimizer']['lr']*rule['gamma']**sum(epoch>=m for m in rule['milestones'])


def make_order(train,canonical,variant,epoch,p):
    if variant=='bfs64_ae':
        roots,metadata=mix_bfs_chunks(canonical,p['batch_size'],p['bfs_chunk_size'],p['seed'],epoch)
    else:
        require(variant in ('gids_ae','random_ae'),'Unknown order')
        roots=np.random.default_rng(np.random.SeedSequence([p['seed'],epoch])).permutation(train)
        metadata=dict(policy='random',seed=p['seed'],epoch=epoch)
    require(roots.dtype==np.int64 and np.array_equal(np.sort(roots),train),'Order lost or duplicated roots')
    return roots,dict(metadata,ordered_nodes_sha256=digest(roots))


def prepare_orders(train,variant,smoke,p):
    provenance=read(HERE/'provenance.json')
    path=Path(provenance['canonical_order']['path'])
    require(sha(path)==provenance['canonical_order']['sha256'],'Canonical BFS changed')
    canonical=np.load(path,allow_pickle=False)
    require(np.array_equal(np.sort(canonical),train),'BFS is not the official train split')
    if smoke:
        train=np.sort(np.random.default_rng(0).permutation(train)[:p['smoke_examples']])
        canonical=canonical[np.isin(canonical,train)]
    n=p['smoke_epochs'] if smoke else p['epochs']
    return [make_order(train,canonical,variant,e,p) for e in range(n)]
