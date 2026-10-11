"""Strict edge checks and an independent oracle for legal whole-group no-fit.

Only the outer grouped layer may underfill. A deficit is accepted only when
the exact frozen RNG/occurrence rule reproduces the emitted neighbor multiset
and counts, with no remaining unit fitting the budget. Inner DGL layers retain
the exact min(degree, fanout) requirement. No sampler or CUDA binary is changed.
"""
from collections import Counter
import numpy as np

MASK=(1<<64)-1


class SamplingAuditError(RuntimeError):
    def __init__(self,message,details):
        super().__init__(message);self.details=details


def demand(condition,message,**details):
    if not condition:raise SamplingAuditError(message,details)


def whole_group_reference(owner,units,nodes,group_size,fanout,seed,position,members,mapping):
    """CPU reconstruction of frozen lane-strided weighted occurrence selection."""
    def splitmix(x):
        x=(x+0x9e3779b97f4a7c15)&MASK
        x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
        x=((x^(x>>27))*0x94d049bb133111eb)&MASK
        return x^(x>>31)
    units=np.asarray(units,dtype=np.int64)
    costs=np.where(units<nodes,1,group_size)
    selected=[];sources=[];groups=0;remaining=fanout
    while remaining>0:
        eligible=[i for lane in range(32) for i in range(lane,len(units),32)
                  if i not in selected and costs[i]<=remaining]
        if not eligible:break
        counter=int(seed)^((int(owner)*0xd6e8feb86659fd93)&MASK)^((int(position)*0xa0761d6478bd642f)&MASK)^len(selected)
        threshold=splitmix(counter)%sum(int(costs[i]) for i in eligible)
        for i in eligible:
            if threshold<int(costs[i]):break
            threshold-=int(costs[i])
        selected.append(i);remaining-=int(costs[i]);unit=int(units[i])
        if unit<nodes:sources.append(unit)
        else:
            gid=int(mapping[unit-nodes]);sources.extend(int(x) for x in members[gid]);groups+=1
    eligible_left=[i for i in range(len(units)) if i not in selected and costs[i]<=remaining]
    return dict(sources=sources,groups=groups,count=len(sources),remaining=remaining,
        selected_occurrences=selected,eligible_left=eligible_left,
        raw_units=int(np.count_nonzero(units<nodes)),group_units=int(np.count_nonzero(units>=nodes)))


def verify_digit_blocks(blocks,arrays,fanouts,bundle,invocation_seed):
    import dgl
    ptr,idx=arrays;proofs=[]
    demand(len(blocks)==len(fanouts),'Wrong block count')
    n=int(bundle.manifest['dataset']['num_nodes']);g=int(bundle.manifest['grouping']['group_size'])
    for layer,(b,fanout) in enumerate(zip(blocks,fanouts)):
        u,v=b.edges(order='eid')
        src=b.srcdata[dgl.NID][u].cpu().numpy();local_dst=v.cpu().numpy()
        dst=b.dstdata[dgl.NID][v].cpu().numpy();eids=b.edata[dgl.EID].cpu().numpy()
        demand(np.all((eids>=0)&(eids<len(idx))),'EID outside original CSC',layer=layer)
        demand(np.array_equal(src,idx[eids]) and np.array_equal(dst,np.searchsorted(ptr,eids,side='right')-1),'Sampled nonexistent edge',layer=layer)
        owners=b.dstdata[dgl.NID].cpu().numpy();degrees=b.in_degrees().cpu().numpy()
        expected=np.minimum(ptr[owners+1]-ptr[owners],fanout)
        if layer!=0:
            demand(np.array_equal(degrees,expected),'Inner uniform sample cardinality mismatch',layer=layer,
                owners=owners[degrees!=expected][:32].tolist(),actual=degrees[degrees!=expected][:32].tolist(),expected=expected[degrees!=expected][:32].tolist())
            continue
        sampled_nodes=b.dstdata['digit_sampled_nodes'].cpu().numpy()
        sampled_groups=b.dstdata['digit_sampled_groups'].cpu().numpy()
        demand(np.array_equal(degrees,sampled_nodes),'Grouped node counters differ from emitted edges',layer=layer)
        demand(np.all((sampled_groups>=0)&(sampled_groups*g<=degrees)),'Invalid grouped counters',layer=layer)
        demand(np.all(degrees<=expected),'Grouped sampling exceeded budget/degree',layer=layer)
        for position in np.flatnonzero(degrees<expected):
            owner=int(owners[position]);rp=bundle.arrays['reordered_indptr']
            units=bundle.arrays['reordered_indices'][rp[owner]:rp[owner+1]]
            ref=whole_group_reference(owner,units,n,g,fanout,invocation_seed,int(position),bundle.arrays['group_members'],bundle.arrays['supernode_to_group'])
            actual=src[local_dst==position].tolist()
            detail=dict(layer=layer,owner=owner,seed_position=int(position),invocation_seed=int(invocation_seed),fanout=int(fanout),
                expected=int(expected[position]),actual=int(degrees[position]),reported_groups=int(sampled_groups[position]),
                actual_sources=actual,reference=ref)
            demand(ref['raw_units']+g*ref['group_units']==int(ptr[owner+1]-ptr[owner]),'Expanded group degree differs from original graph',**detail)
            demand(0<ref['remaining']<g and not ref['eligible_left'],'Deficit is not a whole-group no-fit stop',**detail)
            demand(ref['count']==int(degrees[position]) and ref['groups']==int(sampled_groups[position]) and Counter(actual)==Counter(ref['sources']),
                'Grouped deficit differs from independent CPU reference',**detail)
            proofs.append(dict(detail,reason='whole_group_no_fit',oracle_equal=True))
    return dict(passed=True,whole_group_no_fit=proofs,checked_layers=len(blocks))
