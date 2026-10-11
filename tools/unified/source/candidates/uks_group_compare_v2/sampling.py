"""Original CSC copied into anonymous memory for all three arms."""
import os
from .common import DATA,require

def graph_sampler(arm,seed,variant):
    require(arm in ('gids','digit'),'Unknown arm')
    from candidates.uks_mixed_1k2k_v1.runtime import setup_sampling_imports
    setup_sampling_imports()
    import numpy as np,torch,dgl,digit.sampler as module
    from candidates.uks_native_v1.binding import protocol,bundle,check
    from candidates.uks_group_incremental_v1.common import extension
    from .common import v1,v2
    p=protocol();binding=check();n=p['nodes'];e=binding['receipts']['csc']['normalized_edges']
    # Use anonymous copies for BOTH control and overlay before CUDA pinning.
    # Do not inherit the old file-backed CSC registration route.
    arrays=[]
    for name,count in (('indptr',n+1),('indices',e),('eids',e)):
        src=np.load(DATA/'csc'/(name+'.npy'),mmap_mode='r')
        require(src.dtype==np.int64 and src.shape==(count,),'Wrong original CSC')
        dst=np.empty(count,dtype=np.int64)
        for lo in range(0,count,1048576):dst[lo:lo+1048576]=src[lo:lo+1048576]
        require(dst.flags.owndata and not isinstance(dst,np.memmap),'CSC is not anonymous');arrays.append(dst)
    tensors=tuple(torch.from_numpy(a) for a in arrays)
    graph=dgl.graph(('csc',tensors),num_nodes=n,idtype=torch.int64).formats('csc')
    graph.create_formats_()
    require(all(a.data_ptr()==b.data_ptr() for a,b in zip(tensors,graph.adj_tensors('csc'))),'CSC unexpectedly copied')
    graph.pin_memory_();require(graph.is_pinned(),'Anonymous CSC not pinned')
    if arm=='gids':
        return graph,arrays,None,dgl.dataloading.NeighborSampler(p['fanouts'])
    artifact=bundle()
    if variant=='v1':
        from candidates.uks_group_reuse_v1.overlay import load
        artifact=load(artifact,v1.DEFAULT_OUTPUT)
    elif variant=='v2':
        from candidates.uks_group_reuse_v2.overlay import load
        artifact=load(artifact,v2.DEFAULT_OUTPUT)
    else:raise RuntimeError('Unknown DiGiT overlay')
    os.environ['UKS_GROUP_KERNEL']='incremental';module._cuda_extension=extension()
    sampler=module.DiGiTNeighborSampler(p['fanouts'],artifact,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=seed)
    sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    return graph,arrays,artifact,sampler
