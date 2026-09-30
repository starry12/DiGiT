"""Shared graph/sampler setup for independent profiling and fresh measured workers."""
import gzip
import hashlib
import random
from pathlib import Path
import numpy as np
from .common import ROOT, require


def setup_sampling_imports():
    from ae.pa_sage.common import setup_imports
    setup_imports()
    import sys
    sys.path.insert(0,str(ROOT/'candidates/pa_sage_bidir_native_v2/runtime'))


def startup():
    import torch,dgl
    torch.set_num_threads(16)
    count=dgl.utils.get_num_threads()
    require(count == 16,'Unexpected DGL thread count')
    try:dgl.utils.set_num_threads(1);dgl.seed(0)
    finally:dgl.utils.set_num_threads(count)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def seed(value):
    import torch,dgl
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value);dgl.seed(value)


def graph_sampler(p,binding,value):
    import torch
    from digit import DiGiTSamplerCUDA
    from digit.sampler import DiGiTNeighborSampler
    from .binding import load_bundle
    from candidates.pa_sage_bidir_native_v2.graph_io import load_pinned_csc
    require(Path(DiGiTSamplerCUDA.__file__).resolve() == ROOT/'candidates/pa_sage_bidir_native_v2/runtime/digit/DiGiTSamplerCUDA.so',
            'Wrong frozen sampler binary')
    bundle=load_bundle(p,binding)
    graph,arrays=load_pinned_csc(ROOT/p['data'],p['graph']['nodes'],p['graph']['edges'])
    sampler=DiGiTNeighborSampler(p['fanouts'],bundle,cuda_mode='required',metadata_mode=p['metadata_mode'],random_seed=value)
    sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    return graph,arrays,bundle,sampler


def train_ids(binding):
    desc=binding['train_split']
    with gzip.open(desc['path'],'rt') as f: values=np.fromiter((int(x) for x in f if x.strip()),dtype=np.int64)
    require(len(values)==desc['count'] and len(np.unique(values))==len(values),'Invalid training split')
    return values


def model(p):
    import torch
    from models import SAGE
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    seed(p['seed']); net=SAGE(128,p['hidden'],p['classes'],num_layers=p['layers'],dropout=p['dropout']).cuda()
    kwargs=dict(p['optimizer']['kwargs']);kwargs['betas']=tuple(kwargs['betas'])
    optimizer=torch.optim.Adam(net.parameters(),**kwargs)
    initial=model_hash(net);seed(p['seed']);net.train()
    return net,optimizer,initial
