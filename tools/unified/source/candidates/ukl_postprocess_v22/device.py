"""GPU numbering/block construction with frozen CPU validation retained.

This is deliberately not an entirely GPU-resident pipeline: the frozen sampler
still checks owners and outputs on CPU, and the existing I/O adapter takes CPU
addresses. Only sampled tensors are allocated on GPU, never the full graph.
"""
import ctypes as C
from dataclasses import dataclass
import numpy as np
import torch
from candidates.ukl_native_sampling_v10 import sampling as S

RETAINED = []


@dataclass
class Layer:
    targets: torch.Tensor
    nodes: torch.Tensor
    sources_local: torch.Tensor
    destinations_local: torch.Tensor
    first_source: torch.Tensor
    result: dict


def number_layer(targets, result):
    src, dst = result['src'], result['dst']
    if (targets.ndim != 1 or src.ndim != 1 or dst.shape != src.shape
            or targets.dtype not in (torch.int32,torch.int64)
            or src.dtype not in (torch.int32,torch.int64)
            or dst.dtype not in (torch.int32,torch.int64)
            or not targets.device == src.device == dst.device
            or len(src) > S.MAX_OUTPUT_EDGES or len(targets) > S.MAX_SEEDS):
        raise ValueError('Bounded device integer layer required')
    joined=torch.cat((targets,src)).long()
    unique,inverse=torch.unique(joined,sorted=True,return_inverse=True)
    first=torch.full((len(unique),),len(joined),dtype=torch.int64,device=joined.device)
    first.scatter_reduce_(0,inverse,torch.arange(len(joined),device=joined.device),reduce='amin',include_self=True)
    order=torch.argsort(first);rank=torch.empty_like(order)
    rank[order]=torch.arange(len(order),device=joined.device)
    local=rank[inverse]
    if not torch.equal(local[:len(targets)],torch.arange(len(targets),device=joined.device)):
        raise ValueError('Duplicate targets')
    nodes=joined[first[order]];u=local[len(targets):]
    pos=torch.searchsorted(unique,dst.long())
    if torch.any(pos>=len(unique)).item() or not torch.equal(unique[pos],dst.long()):
        raise ValueError('Unknown destination')
    v=rank[pos]
    if torch.any(v>=len(targets)).item():raise ValueError('Destination outside targets')
    first_source=torch.full((len(nodes),),len(src),device=joined.device,dtype=torch.int64)
    first_source.scatter_reduce_(0,u,torch.arange(len(src),device=joined.device),reduce='amin',include_self=True)
    return Layer(targets.long(),nodes,u,v,first_source,result)


def make_block(layer):
    import dgl
    b=dgl.create_block((layer.sources_local,layer.destinations_local),
                       num_src_nodes=len(layer.nodes),num_dst_nodes=len(layer.targets),device=layer.nodes.device)
    b.srcdata[dgl.NID]=layer.nodes;b.dstdata[dgl.NID]=layer.targets
    b.edata[dgl.EID]=layer.result['eid'].long()
    return b


def storage_rows(layer,primary,storage_extent):
    addresses=primary.long().clone()
    present=layer.first_source<len(layer.result['src'])
    addresses[present]=layer.result['rows'][layer.first_source[present]]
    if torch.any(addresses<0).item() or torch.any(addresses>=storage_extent).item():
        raise RuntimeError('Block storage rows outside graph')
    return addresses


class Native(S.Native):
    """Same frozen sampler/validators, using reusable Torch-owned CUDA outputs."""
    def __init__(self,graph,registration=None):
        super().__init__(graph,registration)
        self.buffers=None;self.last_output=None

    def _cuda(self,view,seeds,fanout,grouped,seed,out):
        try:
            if self.buffers is None:
                self.buffers={'seeds':torch.empty(S.MAX_SEEDS,dtype=torch.int32,device='cuda:0')}
                for k,v in out.items():
                    self.buffers[k]=torch.empty(S.MAX_SEEDS if k in ('counts','errors') else S.MAX_OUTPUT_EDGES,
                                               dtype=torch.int64 if v.dtype==np.int64 else torch.int32,device='cuda:0')
            sp=self.buffers['seeds'][:len(seeds)]
            sp.copy_(torch.from_numpy(seeds))
            output={k:self.buffers[k][:len(v)] for k,v in out.items()}
            for v in output.values():v.fill_(-1)
            # The frozen native entry uses its own default stream; explicitly
            # finish Torch writes before crossing that stream boundary.
            self.registration.api.synchronize()
            code=self.fn(view,C.c_void_p(sp.data_ptr()),len(seeds),fanout,int(grouped),seed,
                         S.Out(*[v.data_ptr() for v in output.values()]))
            self.registration.api.synchronize()
            for k,v in out.items():v[:]=output[k].cpu().numpy()
            self.last_output=output
            return code
        except BaseException:
            self.retained=True;RETAINED.append(self);raise

    def close(self):
        if self.retained:raise RuntimeError('Retained CUDA buffers require graph ownership until exit')
        if self.buffers is not None:
            try:self.registration.api.synchronize()
            except BaseException:
                self.retained=True;RETAINED.append(self);raise
        self.last_output=self.buffers=None
        super().close()


class Sampler(S.Sampler):
    def layers(self,roots,batch=0,audit=None):
        roots=np.asarray(roots)
        if (roots.ndim!=1 or roots.dtype.kind not in 'iu' or not 0<len(roots)<=1024
                or np.any(roots<0) or np.any(roots>=self.native.graph.nodes)
                or len(np.unique(roots))!=len(roots) or type(batch)is not int or batch<0
                or self.seed+batch*3+2>2**64-1):raise ValueError('Invalid roots/batch')
        seeds=np.array(roots,dtype=np.int32,copy=True);layers=[]
        for layer in (2,1,0):
            f=(10,5,5)[layer]
            raw=self.native.sample(seeds,f,grouped=self.grouped and layer==0,
                                   seed=self.seed+batch*3+layer,packed=self.grouped)
            if audit is not None:audit(layer,seeds.copy(),f,self.grouped and layer==0,self.grouped,raw)
            if self.native.gpu:
                if type(self.native)is not Native:raise TypeError('Device postprocessing requires reusable device output owner')
                outputs=self.native.last_output
            else:
                outputs={k:torch.from_numpy(v) for k,v in raw.items()} # CPU oracle tests only
            dev=outputs['src'].device
            mask=torch.arange(f,device=dev)[None,:]<outputs['counts'][:,None]
            result={k:outputs[k].reshape(-1,f)[mask] for k in ('src','dst','eid','rows')}
            numbered=number_layer(torch.as_tensor(seeds,device=dev),result)
            layers.insert(0,numbered)
            # Still required by the unchanged CPU ownership and EID validators.
            seeds=numbered.nodes.cpu().numpy().astype(np.int32,copy=True)
        return seeds.astype(np.int64),layers

    def sample_blocks(self,roots,batch=0,*,layers=None):
        from candidates.ukl_sage_compact_v1.sampler import STORAGE_ROW
        nodes,layers=self.layers(roots,batch) if layers is None else layers
        blocks=[]
        for i,layer in enumerate(layers):
            b=make_block(layer)
            if i==0:
                primary=self.native.graph.arrays['primary'][nodes] if self.grouped else nodes
                b.srcdata[STORAGE_ROW]=storage_rows(layer,torch.as_tensor(np.array(primary,copy=True),device=layer.nodes.device),
                                                  self.native.graph.storage_rows if self.grouped else self.native.graph.nodes)
            blocks.append(b)
        return torch.from_numpy(nodes),torch.from_numpy(np.array(roots,dtype=np.int64,copy=True)),blocks
