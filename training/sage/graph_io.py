"""Read immutable CSC files through writable private mappings for CUDA registration.

NumPy mode c is MAP_PRIVATE copy-on-write: registration can pin writable pages,
but neither CPU nor GPU writes can be persisted to the source NPY files.
Keep the arrays alive for the complete DGL graph lifetime.
"""
import numpy as np
from training.sage.common import require

def load_pinned_csc(data,nodes,edges):
    import torch,dgl
    arrays=[np.load(data/('original_'+name+'.npy'),mmap_mode='c') for name in ('indptr','indices','eids')]
    for a,size in zip(arrays,(nodes+1,edges,edges)):
        require(a.dtype==np.int64 and a.shape==(size,) and a.flags.c_contiguous and a.flags.writeable and a.mode=='c','Invalid private CSC mapping')
    tensors=[torch.from_numpy(a) for a in arrays]
    graph=dgl.graph(('csc',tuple(tensors)),num_nodes=nodes,idtype=torch.int64).formats('csc')
    graph.create_formats_()
    require(graph.num_edges()==edges and all(x.data_ptr()==y.data_ptr() for x,y in zip(tensors,graph.adj_tensors('csc'))),'DGL changed/copied selected CSC')
    graph._graph.pin_memory_()
    require(graph.is_pinned() and all(t.is_pinned() for t in graph.adj_tensors('csc')),'CSC registration failed')
    return graph,arrays
