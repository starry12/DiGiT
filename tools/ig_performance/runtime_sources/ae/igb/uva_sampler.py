"""Synchronous, bounded prototype; original graph and nine metadata arrays on CPU.

No production backend registration. Native first-CSC-occurrence EID semantics.
Every custom kernel is synchronized before returning so pinned owners cannot
be released while a CUDA stream still reads them. This is not a timing path.
"""
import numpy as np
import torch
import dgl
import IGGroupWarpCUDA as native
from bounded_io import source, DEFAULT_CHUNK
from digit.sampler import (DiGiTNeighborSampler, _DIGIT_EDGE_STORAGE_ROW,
                           _DIGIT_EDGE_IS_GROUP)

NAMES = ('reorganized_indptr','reorganized_indices','group_members','group_storage_base',
         'supernode_to_group','node_to_primary','original_indptr','original_indices','original_eids')
I64 = {'reorganized_indptr','original_indptr','original_eids'}
ORIGINAL = NAMES[-3:]


def host_array(value):
    if torch.is_tensor(value):
        if value.device.type != 'cpu':
            raise ValueError('metadata input must reside on CPU')
        value=value.detach().numpy()
    value=np.asarray(value)
    if value.dtype.kind not in 'iu':
        raise ValueError('integer metadata required')
    for at in range(0,value.size,DEFAULT_CHUNK//8):
        part=value.flat[at:at+DEFAULT_CHUNK//8]
        if part.size and (int(part.min())<0 or int(part.max())>2**63-1):
            raise ValueError('metadata outside nonnegative signed int64')
    return value


def validate_metadata(arrays,n,ng,g,storage_rows,compact=True,host_cap=256*1024**2,
                      chunk_bytes=DEFAULT_CHUNK,shared=False):
    if set(arrays)!=set(NAMES) or min(n,g,storage_rows)<=0 or ng<0:
        raise ValueError('invalid metadata dimensions/names')
    arrays={k:source(v) for k,v in arrays.items()}
    shapes={'group_members':(ng,g),'group_storage_base':(ng,),
            'supernode_to_group':(ng,),'node_to_primary':(n,),
            'original_indptr':(n+1,)}
    for k,shape in shapes.items():
        if arrays[k].shape!=shape:
            raise ValueError('wrong shape: '+k)
    # Shape/byte admission happens before reading or allocating any array.
    size=sum(a.size*(8 if not compact or k in I64 or shared and k in ORIGINAL else 4) for k,a in arrays.items())
    if size>host_cap:
        raise ValueError('prototype host metadata cap exceeded before pinning')
    for prefix in ('original','reorganized'):
        ptr,idx=arrays[prefix+'_indptr'],arrays[prefix+'_indices']
        if ptr.ndim!=1 or idx.ndim!=1 or ptr.size<n+1:
            raise ValueError('invalid CSC shape/terminal offset')
        ptr.validate(monotone=True,terminal=idx.size,chunk_bytes=chunk_bytes)
    if arrays['original_eids'].shape!=arrays['original_indices'].shape:
        raise ValueError('EID length differs from CSC indices')
    limits={'reorganized_indices':n+ng,'group_members':n,'group_storage_base':storage_rows-g+1,
            'supernode_to_group':ng,'node_to_primary':storage_rows,'original_indices':n}
    for k,a in arrays.items():
        if k.endswith('indptr'):continue
        limit=limits.get(k,2**63)
        if compact and k not in I64 and not (shared and k in ORIGINAL):limit=min(limit,2**31)
        a.validate(limit=limit,chunk_bytes=chunk_bytes)
    return arrays,size


class PinnedBuffer:
    def __init__(self,array,dtype,device,chunk_bytes=DEFAULT_CHUNK):
        array=source(array)
        self.device=device
        self.length=int(array.size)
        self.dtype=dtype
        # Empty arrays still own a valid registered address; kernels never read it.
        self.storage=torch.empty(max(1,self.length),dtype=dtype,pin_memory=True)
        self.source_sha256=array.copy_to(self.storage[:self.length],chunk_bytes,
                                         limit=2**31 if dtype==torch.int32 else 2**63)
        self.copy_stats=dict(array.stats)
        self.pointer=native.mapped_pointer(self.storage.data_ptr())
        self.nbytes=self.storage.numel()*self.storage.element_size()
        self.owned_bytes=self.nbytes
        self.borrowed_bytes=0

    def data_ptr(self):
        return self.pointer['device_pointer']

    def __getitem__(self,ids):
        if ids.device!=self.device or ids.dtype!=torch.int64 or ids.ndim!=1 or not ids.is_contiguous():
            raise ValueError('gather requires contiguous int64 IDs on bound device')
        if ids.numel() and (ids.min().item()<0 or ids.max().item()>=self.length):
            raise ValueError('mapping lookup out of range')
        out=torch.empty_like(ids)
        fn=native.gather_i32 if self.dtype==torch.int32 else native.gather_i64
        with torch.cuda.device(self.device):
            fn(self.data_ptr(),ids.data_ptr(),out.data_ptr(),ids.numel(),torch.cuda.current_stream(self.device).cuda_stream)
            torch.cuda.current_stream(self.device).synchronize()
        return out


class BorrowedCSCBuffer(PinnedBuffer):
    """Graph owns registration; this object retains both graph and tensor.

    Do not mutate/unpin the graph while bound. Supported unpin/reallocation is
    checked before every kernel; no content rehash is done on the timing path.
    """
    def __init__(self,tensor,graph,device):
        if tensor.device.type!='cpu' or tensor.dtype!=torch.int64 or not tensor.is_contiguous():
            raise ValueError('original CSC must be contiguous CPU int64')
        self.owner=graph
        self.device=device
        self.length=tensor.numel()
        self.dtype=tensor.dtype
        self.storage=tensor
        self.borrowed_bytes=self.length*8
        self.owned_bytes=0
        # Empty DGL tensors have null addresses; a never-read sentinel suffices.
        if not self.length:
            self.storage=torch.empty(1,dtype=torch.int64,pin_memory=True)
            self.owned_bytes=8
        self.pointer=native.mapped_pointer(self.storage.data_ptr())
        self.nbytes=self.borrowed_bytes+self.owned_bytes
        self.copy_stats=dict(max_read_bytes=0,max_copy_elements=0,traversals=0)
        self.source_sha256=None


class UVAMetadata(dict):
    def __init__(self,arrays,n,ng,g,storage_rows,device,compact=True,host_cap=256*1024**2,
                 graph=None,chunk_bytes=DEFAULT_CHUNK):
        super().__init__()
        self.graph=graph
        self.graph_csc=None
        if graph is not None:
            if graph.device.type!='cpu' or not graph.is_pinned() or graph.ndata or graph.edata or graph.num_nodes()!=n:
                raise ValueError('shared original CSC requires matching pinned featureless CPU graph')
            self.graph_csc=graph.adj_tensors('csc')
            for name,tensor in zip(ORIGINAL,self.graph_csc):
                value=arrays[name]
                if not torch.is_tensor(value) or value.dtype!=torch.int64 or value.data_ptr()!=tensor.data_ptr() or value.shape!=tensor.shape:
                    raise ValueError('shared CSC input must be the exact graph storage')
        arrays,logical_bytes=validate_metadata(arrays,n,ng,g,storage_rows,compact,host_cap,chunk_bytes,graph is not None)
        self.n,self.ng,self.g=n,ng,g
        self.device=torch.device(device)
        if self.device.type!='cuda' or self.device.index is None:
            raise ValueError('indexed CUDA device required')
        if native.UVA_EID64_API!=1 or native.SHARED_CSC_API!=1:
            raise ValueError('prototype extension API mismatch')
        self.compact=compact
        self.closed=False
        with torch.cuda.device(self.device):
            for name,array in arrays.items():
                dtype=torch.int64 if not compact or name in I64 else torch.int32
                self[name]=(BorrowedCSCBuffer(self.graph_csc[ORIGINAL.index(name)],graph,self.device)
                            if graph is not None and name in ORIGINAL else PinnedBuffer(array,dtype,self.device,chunk_bytes))
        self.accounting=dict(metadata_gpu_tensor_bytes=0,host_logical_bytes=logical_bytes,
                             host_pinned_allocation_bytes=sum(v.nbytes for v in self.values()),
                             owned_pinned_bytes=sum(v.owned_bytes for v in self.values()),
                             borrowed_csc_bytes=sum(v.borrowed_bytes for v in self.values()),
                             original_csc_owned_bytes=sum(self[k].owned_bytes for k in ORIGINAL),
                             arrays={k:dict(dtype=str(v.dtype),elements=v.length,owned_bytes=v.owned_bytes,
                                       borrowed_bytes=v.borrowed_bytes,copy_stats=v.copy_stats,
                                       source_sha256=v.source_sha256,**v.pointer) for k,v in self.items()},
                             compact_ids=compact,eid_bits=64,synchronous_prototype=True,
                             shared_original_csc=graph is not None,chunk_bytes=chunk_bytes)

    def check_graph(self):
        if self.graph is None:return
        if not self.graph.is_pinned():raise ValueError('shared graph was unpinned while bound')
        current=self.graph.adj_tensors('csc')
        if any(a.data_ptr()!=b.data_ptr() or a.shape!=b.shape for a,b in zip(current,self.graph_csc)):
            raise ValueError('shared CSC storage changed while bound')

    def sample(self,seeds,fanout,random_seed):
        if self.closed:
            raise ValueError('metadata owner is closed')
        self.check_graph()
        if seeds.device!=self.device or seeds.dtype!=torch.int64 or seeds.ndim!=1 or not seeds.is_contiguous():
            raise ValueError('seeds must be contiguous CUDA int64 on bound device')
        if not self.g<=fanout<=native.MAX_FANOUT or not 0<=random_seed<2**64:
            raise ValueError('invalid fanout/random seed')
        if seeds.numel()>1048576:
            raise ValueError('prototype seed batch cap exceeded')
        if seeds.numel() and (seeds.min().item()<0 or seeds.max().item()>=self.n):
            raise ValueError('seed outside graph')
        count=seeds.numel()*fanout
        sources=torch.empty(count,dtype=torch.int64,device=self.device)
        rows=torch.empty_like(sources)
        flags=torch.empty(count,dtype=torch.uint8,device=self.device)
        eids=torch.empty_like(sources)
        groups=torch.empty(seeds.numel(),dtype=torch.int64,device=self.device)
        nodes=torch.empty_like(groups)
        fn=native.sample_group_aware_i32_eid64 if self.compact else native.sample_group_aware
        if self.compact and self.graph is not None:fn=native.sample_group_aware_shared_i64_csc
        with torch.cuda.device(self.device):
            stream=torch.cuda.current_stream(self.device)
            fn(*[self[k].data_ptr() for k in NAMES],seeds.data_ptr(),seeds.numel(),self.n,self.ng,self.g,
               fanout,random_seed,sources.data_ptr(),rows.data_ptr(),flags.data_ptr(),eids.data_ptr(),
               groups.data_ptr(),nodes.data_ptr(),stream.cuda_stream)
            stream.synchronize()
        if sources.numel() and torch.any((sources>=0)&(eids<0)).item():
            raise ValueError('selected edge absent from original CSC')
        return sources,rows,flags,eids,groups,nodes

    def close(self):
        if not self.closed:
            # All operations are synchronous; no outstanding custom reader remains.
            self.clear()
            self.graph_csc=None
            self.graph=None
            self.closed=True


def graph_from_edges(sources,destinations,n):
    # Fixture helper only. Full graph entry is bounded_io.load_csc.
    if np.size(sources)>1048576 or n>1048576:
        raise ValueError('COO helper is restricted to bounded fixtures; use load_csc')
    src,dst=host_array(sources),host_array(destinations)
    if src.ndim!=1 or dst.shape!=src.shape or n<=0:
        raise ValueError('invalid edge vectors')
    if src.size and max(src.max(),dst.max())>=n:
        raise ValueError('edge endpoint outside graph')
    graph=dgl.graph((torch.from_numpy(src.astype(np.int64,copy=True)),
                     torch.from_numpy(dst.astype(np.int64,copy=True))),num_nodes=n,idtype=torch.int64)
    graph=graph.formats('csc')
    graph.create_formats_()
    if graph.ndata or graph.edata:
        raise ValueError('featureless graph must not bind node/edge data')
    graph.pin_memory_()
    ptr,idx,eid=graph.adj_tensors('csc')
    if ptr.numel()!=n+1 or ptr[-1].item()!=len(idx) or len(eid)!=len(idx):
        raise ValueError('invalid DGL CSC')
    assert graph.device.type=='cpu' and graph.is_pinned()
    return graph


def arrays_from_graph(graph,bundle):
    if graph.device.type!='cpu' or not graph.is_pinned() or graph.ndata or graph.edata:
        raise ValueError('CPU pinned featureless graph required')
    ptr,idx,eid=graph.adj_tensors('csc')
    return dict(reorganized_indptr=bundle.arrays['reordered_indptr'],
                reorganized_indices=bundle.arrays['reordered_indices'],group_members=bundle.arrays['group_members'],
                group_storage_base=bundle.arrays['group_storage_base'],supernode_to_group=bundle.arrays['supernode_to_group'],
                node_to_primary=bundle.arrays['node_to_primary_row'],
                original_indptr=ptr,original_indices=idx,original_eids=eid)


class UVANeighborSampler(DiGiTNeighborSampler):
    def __init__(self,fanouts,bundle,compact=True,random_seed=0,host_cap=256*1024**2,chunk_bytes=DEFAULT_CHUNK):
        super().__init__(fanouts,bundle,cuda_mode='required',metadata_mode='gpu',random_seed=random_seed)
        self.compact=compact
        self._bound_graph=None
        self.host_cap=host_cap
        self.chunk_bytes=chunk_bytes

    def _ensure_cuda_metadata(self,graph,device):
        if self._bound_graph is not None and graph is not self._bound_graph:
            raise ValueError('prototype cannot rebind a graph; create a fresh sampler')
        self._bound_graph=graph
        key=str(device)
        if key not in self._cuda_metadata:
            self._cuda_metadata[key]=UVAMetadata(arrays_from_graph(graph,self.bundle),self.num_nodes,self.num_groups,
                    self.group_size,len(self.bundle.arrays['storage_to_node']),device,self.compact,
                    self.host_cap,graph=graph,chunk_bytes=self.chunk_bytes)
        return self._cuda_metadata[key]

    def _cuda_group_frontier(self,graph,seed_nodes,fanout):
        metadata=self._ensure_cuda_metadata(graph,seed_nodes.device)
        invocation_seed=self.random_seed+self._cuda_call_counter
        sources,rows,flags,eids,groups,nodes=metadata.sample(seed_nodes,fanout,invocation_seed)
        self._cuda_call_counter+=1
        valid=sources>=0
        destinations=seed_nodes.repeat_interleave(fanout)[valid]
        frontier=dgl.graph((sources[valid],destinations),num_nodes=self.num_nodes,idtype=torch.int64,device=seed_nodes.device)
        frontier.edata[dgl.EID]=eids[valid]
        frontier.edata[_DIGIT_EDGE_STORAGE_ROW]=rows[valid]
        frontier.edata[_DIGIT_EDGE_IS_GROUP]=flags[valid].to(torch.bool)
        return frontier,groups,nodes

    def close(self):
        for value in self._cuda_metadata.values():
            value.close()
        self._cuda_metadata.clear()
        self._bound_graph=None
