"""64-bit CSC/EID boundaries, independently of feature-file preparation."""
import numpy as np
from .common import require,heavy_gate


def offsets64(ptr,total_edges):
    require(isinstance(ptr,np.ndarray) and ptr.dtype==np.int64 and ptr.ndim==1 and ptr.flags.c_contiguous,'CSC offsets must remain int64')
    require(type(total_edges) is int and 0<=total_edges<2**63 and len(ptr)>=2 and ptr[0]==0 and ptr[-1]==total_edges,
            'Invalid CSC endpoint offsets')
    # Chunked by caller for production validation; the arithmetic is 64-bit.
    require(np.all(ptr[1:]>=ptr[:-1]),'Nonmonotone CSC offsets')
    return ptr


def edge_ids64(values,total_edges):
    require(isinstance(values,np.ndarray) and values.dtype==np.int64 and values.ndim==1,'EIDs must remain int64')
    require(not len(values) or (values.min()>=0 and values.max()<total_edges),'EID outside graph')
    return values


def compact_ids(values):
    require(values.dtype==np.int64 and values.ndim>=1,'Logical/storage IDs start as int64')
    require(not values.size or (values.min()>=-1 and values.max()<2**31),'Logical/storage IDs cannot fit signed int32')
    return values.astype(np.int32)


def normalized_fixture(edges,nodes):
    require(nodes<=4096 and edges.dtype==np.int64 and edges.ndim==2 and edges.shape[0]==2 and edges.shape[1]<=131072,'Small int64 edge fixture only')
    require(not edges.size or (edges.min()>=0 and edges.max()<nodes),'Bad endpoint')
    src,dst=edges[:,edges[0]!=edges[1]];ids=np.arange(nodes,dtype=np.int64)
    src=np.r_[src,ids];dst=np.r_[dst,ids];order=np.argsort(dst,kind='stable')
    ptr=np.r_[0,np.cumsum(np.bincount(dst,minlength=nodes),dtype=np.int64)]
    return ptr,src[order].copy(),np.arange(len(src),dtype=np.int64)


def graph_from_csc(ptr,idx,eids,nodes,device='cpu',fixture=False):
    if fixture:require(nodes<=4096 and len(idx)<=131072 and device=='cpu','Invalid fixture graph')
    else:heavy_gate()
    offsets64(ptr,len(idx));edge_ids64(eids,len(idx))
    require(len(ptr)==nodes+1 and idx.dtype==np.int64 and len(eids)==len(idx) and
            (not len(idx) or (idx.min()>=0 and idx.max()<nodes)),'Bad original CSC')
    import torch,dgl
    tensors=tuple(torch.from_numpy(a) for a in (ptr,idx,eids))
    graph=dgl.graph(('csc',tensors),num_nodes=nodes,idtype=torch.int64).formats('csc')
    if device=='cuda':graph.pin_memory_()
    else:require(device=='cpu','Unsupported graph device')
    return graph
