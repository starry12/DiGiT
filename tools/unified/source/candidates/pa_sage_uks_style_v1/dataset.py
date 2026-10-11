import numpy as np
from .common import *

def bundle():
    setup()
    from digit.artifacts import load_artifact_bundle
    from candidates.pa_sage_bidir_native_v2.overlay import apply
    p=cfg();base=load_artifact_bundle(ROOT/p['base_layout']/'final/bundle',validation_mode='fast')
    proof=prepared()['reuse'];n=p['graph']['nodes'];g=base.num_groups
    ptr=np.memmap(DATA/'adjacency/ptr.i64',dtype=np.int64,mode='r')
    idx=np.memmap(DATA/'adjacency/units.i64',dtype=np.int64,mode='r')
    require(len(ptr)==n+g+1 and ptr[0]==0 and ptr[n]==ptr[-1]==len(idx),'Invalid overlay extent')
    require(proof['passed'] and proof['expanded_multigraph_exact'],'Missing exact group reuse validation')
    return apply(base,ptr,idx,p['graph']['edges'])

def graph_sampler(arm,seed):
    setup();import torch,dgl
    p=cfg()
    from candidates.pa_sage_ae_512b_v1.graph_io import load_pinned_csc
    graph,arrays=load_pinned_csc(ROOT/p['data'],p['graph']['nodes'],p['graph']['edges'])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);dgl.seed(seed);dgl.random.seed(seed)
    if arm=='gids':return graph,arrays,None,dgl.dataloading.NeighborSampler(p['fanouts'],replace=False)
    from candidates.uks_group_first_v1.sampling import sampler_class
    from .sampling import optimize
    artifact=bundle();sampler=sampler_class(use_cuda=True)(p['fanouts'],artifact,cuda_mode='required',metadata_mode=p['metadata_mode'],random_seed=seed)
    sampler=optimize(sampler);sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    return graph,arrays,artifact,sampler

def roots(epoch=0):
    from candidates.cache_epoch_v1.core import batches
    return np.concatenate(list(batches(np.load(DATA/'canonical.npy'),epoch=epoch)))
