"""Bounded host reference producing only sampled DGL blocks, no full DGL graph."""
import numpy as np
from .common import require,cpu_gate

STORAGE_ROW='digit_storage_row'


def uniform_slots(degree,count,rng):
    require(type(degree) is int and 0<=degree<2**63 and type(count) is int and 0<=count<=1024,'Invalid sampling budget')
    # Partial Fisher-Yates: O(fanout) memory even for degree > 2**32.
    swaps={};chosen=[]
    for i in range(min(count,degree)):
        remaining=degree-i;j=int(rng.integers(0,remaining));chosen.append(swaps.get(j,j))
        swaps[j]=swaps.get(remaining-1,remaining-1)
    return np.asarray(chosen,dtype=np.int64)


def grouped_slots(units,nodes,budget,rng,chunk=65536):
    """Legacy uniform_nodewise_v1 semantics: PPS cost 1/2, no replacement.

    Bounded chunk scans preserve semantics without a degree-sized GPU array.
    This CPU reference trades scanning time for memory; it is not a fast native kernel.
    """
    require(type(budget) is int and 0<=budget<=1024 and 0<chunk<=65536,'Invalid group budget')
    selected=[];used=set();remaining=budget
    while remaining:
        total=0
        for lo in range(0,len(units),chunk):
            costs=np.where(units[lo:lo+chunk]<nodes,1,2)
            costs[costs>remaining]=0
            for i in used:
                if lo<=i<lo+len(costs):costs[i-lo]=0
            total+=int(costs.sum())
        if not total:break
        rank=int(rng.integers(0,total));chosen=None;cost=None
        for lo in range(0,len(units),chunk):
            costs=np.where(units[lo:lo+chunk]<nodes,1,2);costs[costs>remaining]=0
            for i in used:
                if lo<=i<lo+len(costs):costs[i-lo]=0
            part=int(costs.sum())
            if rank>=part:rank-=part;continue
            at=int(np.searchsorted(np.cumsum(costs,dtype=np.int64),rank,side='right'))
            chosen=lo+at;cost=int(costs[at]);break
        require(chosen is not None and cost>0,'Weighted selection failed')
        selected.append(chosen);used.add(chosen);remaining-=cost
    return np.asarray(selected,dtype=np.int64)


def block_from_edges(targets,sources,destinations,eids):
    import torch,dgl
    # Local indices scale with the sampled frontier, not total graph N/E.
    src=list(map(int,targets));lookup={n:i for i,n in enumerate(src)};dest={n:i for i,n in enumerate(src)}
    require(len(lookup)==len(src),'Duplicate roots')
    for n in sources:
        if int(n) not in lookup:lookup[int(n)]=len(src);src.append(int(n))
    u=torch.tensor([lookup[int(n)] for n in sources],dtype=torch.int64)
    v=torch.tensor([dest[int(n)] for n in destinations],dtype=torch.int64)
    b=dgl.create_block((u,v),num_src_nodes=len(src),num_dst_nodes=len(targets))
    b.srcdata[dgl.NID]=torch.tensor(src,dtype=torch.int64);b.dstdata[dgl.NID]=torch.tensor(targets,dtype=torch.int64)
    b.edata[dgl.EID]=torch.as_tensor(np.asarray(eids,dtype=np.int64))
    return b


class HostSampler:
    def __init__(self,graph,fanouts,*,groups=None,seed=0):
        require(graph.fixture and graph.nodes<=4096 and graph.edges<=131072,'CPU reference is fixture-only; native sampler deferred')
        require(len(fanouts)==3 and all(type(f) is int and 0<f<=1024 for f in fanouts),'Invalid fanouts')
        self.graph=graph;self.fanouts=fanouts;self.groups=groups;self.rng=np.random.default_rng(seed)

    def sample_blocks(self,targets):
        import torch,dgl
        cpu_gate();targets=np.asarray(targets,dtype=np.int64)
        require(targets.ndim==1 and len(targets)>0 and targets.min()>=0 and targets.max()<self.graph.nodes,'Bad roots')
        output=targets.copy();seeds=targets.copy();blocks=[]
        for layer in range(len(self.fanouts)-1,-1,-1):
            sources=[];destinations=[];eids=[];preferred={}
            for owner in seeds:
                owner=int(owner)
                if layer==0 and self.groups is not None:
                    units=self.groups.row(owner);slots=grouped_slots(units,self.graph.nodes,self.fanouts[layer],self.rng)
                    selected=[]
                    for unit in units[slots]:
                        members,rows=self.groups.expand(unit);selected.extend(members)
                        for n,row in rows.items():preferred.setdefault(n,row)
                    ids=self.graph.resolve(owner,selected) if selected else np.empty(0,np.int64)
                else:
                    row=self.graph.row(owner);slots=uniform_slots(len(row),self.fanouts[layer],self.rng)
                    selected=row[slots].astype(np.int64).tolist();ids=self.graph.eids(owner,slots)
                sources.extend(selected);destinations.extend([owner]*len(selected));eids.extend(ids.tolist())
            block=block_from_edges(seeds,sources,destinations,eids)
            seeds=block.srcdata[dgl.NID].numpy()
            if layer==0:
                rows=seeds.copy() if self.groups is None else self.groups.primary[seeds].copy()
                for i,n in enumerate(seeds):
                    if int(n) in preferred:rows[i]=preferred[int(n)]
                block.srcdata[STORAGE_ROW]=torch.from_numpy(rows.astype(np.int64,copy=False))
            blocks.insert(0,block)
        return blocks[0].srcdata[dgl.NID],torch.from_numpy(output),blocks
