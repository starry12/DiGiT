"""Owned mapped graph, device frontiers and requests; no whole-graph GPU copy."""
import ctypes as C
import numpy as np
import torch
from .build import verify

STORAGE='digit_fast_storage_row'
HEADS='digit_fast_pair_head'

def number(seeds,result,counts):
    # Preserve targets first and first-occurrence source ordering exactly.
    joined=torch.cat((seeds.long(),result['src'].long()))
    unique,inverse=torch.unique(joined,sorted=True,return_inverse=True)
    first=torch.full((len(unique),),len(joined),dtype=torch.int64,device=joined.device)
    first.scatter_reduce_(0,inverse,torch.arange(len(joined),device=joined.device),reduce='amin',include_self=True)
    order=torch.argsort(first);rank=torch.empty_like(order);rank[order]=torch.arange(len(order),device=joined.device)
    local=rank[inverse];nodes=joined[first[order]];u=local[len(seeds):]
    # Output is row-major by owner: no binary search for destination IDs.
    v=torch.repeat_interleave(torch.arange(len(seeds),device=joined.device),counts.long())
    first_source=torch.full((len(nodes),),len(u),dtype=torch.int64,device=joined.device)
    first_source.scatter_reduce_(0,u,torch.arange(len(u),device=joined.device),reduce='amin',include_self=True)
    if not torch.equal(local[:len(seeds)],torch.arange(len(seeds),device=joined.device)):
        raise ValueError('Duplicate target nodes')
    return nodes,u,v,first_source

def classes(BaseNative,BaseSampler,S,kind):
    class Native(BaseNative):
        def __init__(self,graph,registration=None):
            super().__init__(graph,registration)
            if not self.gpu:return
            self.fast_lib=C.CDLL(str(verify(kind,'sampler')))
            self.fast_fn=self.fast_lib.sample_stream
            self.fast_fn.argtypes=[S.View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,C.c_int64,S.Out,C.c_void_p,C.c_uint64]
            self.fast_fn.restype=C.c_int
            self.address_fn=self.fast_lib.annotate_stream
            self.address_fn.argtypes=[S.View,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int64,C.c_int,C.c_int64,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint64]
            self.address_fn.restype=C.c_int
        def device_sample(self,seeds,fanout,grouped,seed,packed):
            if self.closed or self.retained:raise RuntimeError('Sampler unavailable')
            self.graph._check_owned();self._check_registration()
            if seeds.dtype!=torch.int32 or not seeds.is_cuda or not seeds.is_contiguous() or not 0<len(seeds)<=S.MAX_SEEDS:
                raise ValueError('Bounded device seeds required')
            # Checked kernel may mark a malformed owner; never mutate a previous block.
            checked=seeds.clone();n=len(seeds);size=n*fanout;out={}
            for key in ('src','dst','counts','errors','eid','rows'):
                out[key]=torch.full((n if key in ('counts','errors') else size,),-1,
                    dtype=torch.int64 if key in ('eid','rows') else torch.int32,device=seeds.device)
            groups=torch.zeros(n,dtype=torch.int64,device=seeds.device)
            stream=torch.cuda.current_stream(seeds.device)
            rc=self.fast_fn(self._view(packed),checked.data_ptr(),n,fanout,int(grouped),seed,
                self.graph.storage_rows if packed else self.graph.nodes,S.Out(*[out[k].data_ptr() for k in ('src','dst','counts','errors','eid','rows')]),groups.data_ptr(),stream.cuda_stream)
            if rc:raise RuntimeError('Native sampler launch failed: '+str(rc))
            # One scalar result, not all sampled endpoints/EIDs, crosses to CPU.
            if bool(torch.any(out['errors']!=0)):raise RuntimeError('Native owner/edge/storage validation failed')
            mask=torch.arange(fanout,device=seeds.device)[None,:]<out['counts'][:,None]
            result={k:out[k].reshape(n,fanout)[mask] for k in ('src','dst','eid','rows')}
            return result,out['counts'],groups
        def annotate(self,nodes,first,result,grouped):
            out=torch.empty_like(nodes);heads=torch.empty_like(nodes);errors=torch.zeros(1,dtype=torch.int32,device=nodes.device)
            rc=self.address_fn(self._view(grouped),nodes.data_ptr(),first.data_ptr(),result['rows'].data_ptr(),len(result['rows']),len(nodes),
                self.graph.storage_rows if grouped else self.graph.nodes,int(grouped),out.data_ptr(),heads.data_ptr(),errors.data_ptr(),torch.cuda.current_stream().cuda_stream)
            if rc or int(errors.item()):raise RuntimeError('Device feature alias/group-head validation failed')
            return out,heads

    class Sampler(BaseSampler):
        def sample_blocks(self,roots,batch=0,**kwargs):
            if not self.native.gpu:return super().sample_blocks(roots,batch,**kwargs)
            if kwargs:raise ValueError('Device path builds its own validated layers')
            roots=np.asarray(roots)
            if roots.ndim!=1 or roots.dtype!=np.int64 or not 0<len(roots)<=1024 or len(np.unique(roots))!=len(roots) or roots.min()<0 or roots.max()>=self.native.graph.nodes:
                raise ValueError('Invalid training roots')
            if type(batch)!=int or batch<0 or self.seed+batch*3+2>=2**64:raise ValueError('Batch seed overflow')
            import dgl
            seeds=torch.as_tensor(np.array(roots,copy=True),device='cuda',dtype=torch.int32)
            output=seeds.long();blocks=[]
            for layer in (2,1,0):
                result,counts,groups=self.native.device_sample(seeds,(10,5,5)[layer],self.grouped and layer==0,self.seed+batch*3+layer,self.grouped)
                nodes,u,v,first=number(seeds,result,counts)
                block=dgl.create_block((u,v),num_src_nodes=len(nodes),num_dst_nodes=len(seeds),device=nodes.device)
                block.srcdata[dgl.NID]=nodes;block.dstdata[dgl.NID]=seeds.long();block.edata[dgl.EID]=result['eid']
                if layer==0:
                    rows,heads=self.native.annotate(nodes,first,result,self.grouped)
                    block.srcdata[STORAGE]=rows;block.srcdata[HEADS]=heads
                    self.last_group_counts=dict(grouped_sampled_edges=groups.sum(),outer_sampled_edges=len(u))
                blocks.insert(0,block);seeds=nodes.to(torch.int32)
            return nodes,output,blocks
    return Native,Sampler
