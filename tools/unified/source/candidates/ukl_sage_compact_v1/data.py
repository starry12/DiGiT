"""Small deterministic UKL inputs and complete independent epoch order."""
import numpy as np
from .common import bounded,require,digest


def fixture_inputs(p):
    require(p.get('fixture'),'Only bounded inputs are generated in this version')
    bounded(p['nodes'],0,True);n=p['nodes'];spec=p['synthetic']
    features=np.random.Generator(np.random.PCG64(np.random.SeedSequence([spec['feature_seed'],0]))).random((n,128),dtype=np.float32)*np.float32(2)-np.float32(1)
    labels=np.random.Generator(np.random.PCG64(np.random.SeedSequence([spec['label_seed'],0]))).integers(0,p['classes'],size=n,dtype=np.int64)
    ids=np.sort(np.random.default_rng(p['training']['split_seed']).choice(n,p['training']['train_nodes'],replace=False)).astype(np.int64)
    return features,labels,ids


def epoch_order(p,selected):
    bounded(p['nodes'],0,bool(p.get('fixture')))
    require(selected.dtype==np.int64 and selected.ndim==1 and len(selected)==p['training']['train_nodes'] and
            len(selected)>0 and selected[0]>=0 and selected[-1]<p['nodes'] and np.all(np.diff(selected)>0),'Bad training split')
    roots=selected[np.random.default_rng(p['training']['order_seed']).permutation(len(selected))]
    return roots,dict(examples=len(roots),updates=(len(roots)+p['batch_size']-1)//p['batch_size'],
        last_batch=len(roots)%p['batch_size'] or p['batch_size'],root_sha256=digest(roots))
