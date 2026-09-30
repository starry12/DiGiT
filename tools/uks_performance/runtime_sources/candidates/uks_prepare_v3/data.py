"""Deterministic inputs and complete root order. Large generation stays gated."""
import numpy as np
from .common import require,digest,heavy_gate


def row_block(p,start,count,labels=False):
    if not p.get('fixture'):heavy_gate()
    spec=p['synthetic'];block=spec['block_rows']
    require(0<=start<=p['nodes'] and start%block==0 and 0<=count<=min(block,p['nodes']-start),'Unaligned generation block')
    seed=spec['label_seed' if labels else 'feature_seed']
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,start//block])))
    return rng.integers(0,p['classes'],size=count,dtype=np.int64) if labels else rng.random((count,256),dtype=np.float32)*np.float32(2)-np.float32(1)


def training_ids(p):
    if not p.get('fixture'):heavy_gate()
    n,t=p['nodes'],p['training']['train_nodes'];require(0<t<=n,'Invalid training fraction')
    return np.sort(np.random.Generator(np.random.PCG64(p['training']['split_seed'])).choice(n,size=t,replace=False)).astype(np.int64)


def epoch_order(p,selected):
    require(selected.dtype==np.int64 and selected.ndim==1 and len(selected)==p['training']['train_nodes'],'Wrong split extent')
    if len(selected)>4096:heavy_gate()
    require(selected[0]>=0 and selected[-1]<p['nodes'] and np.all(np.diff(selected)>0),'Split must be sorted, unique and in range')
    roots=selected[np.random.Generator(np.random.PCG64(p['training']['order_seed'])).permutation(len(selected))]
    return roots,dict(examples=len(roots),updates=(len(roots)+p['batch_size']-1)//p['batch_size'],root_sha256=digest(roots),
        last_batch=len(roots)%p['batch_size'] or p['batch_size'],warmup_batches=0)


def batches(roots,size):
    require(type(size) is int and size>0,'Invalid batch size')
    for lo in range(0,len(roots),size):yield roots[lo:lo+size]


def fixture_inputs(p):
    require(p.get('fixture') and p['nodes']<=4096,'Small CPU fixture only')
    return row_block(p,0,p['nodes']),row_block(p,0,p['nodes'],labels=True),training_ids(p)
