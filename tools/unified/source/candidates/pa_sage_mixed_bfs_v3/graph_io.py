"""AE adjacency and validated reverse-edge overlay; immutable source files."""
import copy
import numpy as np
from .common import require,read


def overlay_manifest(binding):
    m=copy.deepcopy(read(binding['bundle']+'/manifest.json'))
    prep=read(binding['prepared'])
    require(prep['passed'] and prep['validation']['passed'],'Unverified AE overlay')
    require(prep['base_manifest_sha256']==binding['bundle_manifest_sha256'],'Overlay base differs')
    m['dataset']['num_edges']=prep['validation']['edges']
    for key in ('reordered_indptr','reordered_indices'):
        a=np.load(binding['overlay'][key],mmap_mode='r')
        require(a.dtype==np.int64 and a.ndim==1,'Wrong overlay array')
        m['files'][key]['shape']=list(a.shape)
    require(m['files']['reordered_indices']['shape']==[prep['validation']['units']],'Wrong overlay extent')
    return m


def load_bundle(binding):
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import load_bundle as base_bundle
    from digit.artifacts import ArtifactBundle
    base=base_bundle(binding);arrays=dict(base.arrays);m=overlay_manifest(binding)
    for key in ('reordered_indptr','reordered_indices'):
        arrays[key]=np.load(binding['overlay'][key],mmap_mode='r')
    m.setdefault('metadata',{})['bidirectional_overlay']=True
    return ArtifactBundle(base.root,m,arrays)


def load_graph(binding,nodes,edges,pin=True):
    import torch,dgl
    # Allocate anonymous arrays before pinning; never register file-backed pages.
    arrays=[]
    for key,count in (('indptr',nodes+1),('indices',edges)):
        source=np.load(binding['csc'][key],mmap_mode='r')
        require(source.dtype==np.int64 and source.shape==(count,),'Wrong AE CSC extent')
        target=np.empty(count,dtype=np.int64)
        for lo in range(0,count,1048576):target[lo:lo+1048576]=source[lo:lo+1048576]
        require(target.flags.owndata and not isinstance(target,np.memmap),'CSC is not anonymous')
        arrays.append(target)
    require(arrays[0][0]==0 and arrays[0][-1]==edges,'Bad CSC boundaries')
    eids=torch.arange(edges,dtype=torch.int64)
    tensors=(torch.from_numpy(arrays[0]),torch.from_numpy(arrays[1]),eids)
    graph=dgl.graph(('csc',tensors),num_nodes=nodes,idtype=torch.int64).formats('csc')
    require(graph.num_edges()==edges and all(x.data_ptr()==y.data_ptr() for x,y in zip(tensors,graph.adj_tensors('csc'))),'CSC copied or changed')
    if pin:
        graph._graph.pin_memory_()
        require(graph.is_pinned() and all(t.is_pinned() for t in graph.adj_tensors('csc')),'Anonymous CSC registration failed')
    return graph,arrays
