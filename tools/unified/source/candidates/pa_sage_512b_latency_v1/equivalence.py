"""Numerical and structural oracles, run outside all performance intervals."""
import copy
import torch
import dgl
from .model import FunctionalMeanSAGE, functional_layer
from models import SAGE
from .frontier import coo_frontier


def close(a,b,record):
    if a is None or b is None:
        if a is not b:raise AssertionError('Gradient presence differs')
        return
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    if a.numel():
        record['max_abs_error']=max(record['max_abs_error'],float((a-b).abs().max().item()))
        record['bit_exact']=record['bit_exact'] and torch.equal(a,b)
    record['comparisons']+=1


def frontier_equivalence(device):
    cases=[([],[],0),([],[],8),([0,1,1,3,3,7],[0,0,0,3,2,7],8),
           ([7,0,4,7],[0,7,4,0],8)]
    for us,vs,n in cases:
        u=torch.tensor(us,dtype=torch.int64,device=device);v=torch.tensor(vs,dtype=torch.int64,device=device)
        old=dgl.graph((u,v),num_nodes=n,idtype=torch.int64,device=device)
        new=coo_frontier(u,v,n,check_bounds=True)
        assert old.ntypes==new.ntypes and old.etypes==new.etypes
        assert old.num_nodes()==new.num_nodes() and old.num_edges()==new.num_edges()
        for fmt in ('coo','csc','csr'):
            for a,b in zip(old.adj_tensors(fmt),new.adj_tensors(fmt)):assert torch.equal(a,b)
        if n:
            # Every destination is in the seed list, preserving repeated edges.
            seeds=torch.arange(n,dtype=torch.int64,device=device)
            ob=dgl.to_block(old,seeds);nb=dgl.to_block(new,seeds)
            for a,b in zip(ob.edges(order='eid'),nb.edges(order='eid')):assert torch.equal(a,b)
            for key in (dgl.NID,):
                assert torch.equal(ob.srcdata[key],nb.srcdata[key])
                assert torch.equal(ob.dstdata[key],nb.dstdata[key])
            assert torch.equal(ob.edata[dgl.EID],nb.edata[dgl.EID])
    return dict(passed=True,cases=len(cases),adjacency_and_block_bit_exact=True)


def blocks_for(device,case,idtype):
    blocks=[]
    for nsrc,ndst in ((8,6),(6,4),(4,3)):
        if case=='empty':u=[];v=[]
        else:
            pairs=[(j,i) for i in range(ndst-1) for j in range(nsrc) if (j+i)%3==0]
            pairs += [(0,0),(0,0),(1,1)]
            u,v=map(list,zip(*pairs))
        blocks.append(dgl.create_block((torch.tensor(u,dtype=idtype,device=device),
            torch.tensor(v,dtype=idtype,device=device)),num_src_nodes=nsrc,num_dst_nodes=ndst,
            idtype=idtype,device=device))
    return blocks


def rng_state(device):
    return (torch.get_rng_state(),torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None)


def restore_rng(state,device):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state(state[1],device)


def model_equivalence(device):
    result=dict(passed=True,bit_exact=True,max_abs_error=0.,comparisons=0,model_updates=0,
                rng_exact=True,layer_cases=0)
    for case in ('empty','mixed'):
        for idtype in (torch.int32,torch.int64):
            blocks=blocks_for(device,case,idtype)
            torch.manual_seed(23);original=SAGE(8,8,12,3,.2).to(device)
            torch.manual_seed(23);candidate=FunctionalMeanSAGE(8,8,12,3,.2).to(device)
            assert list(original.state_dict())==list(candidate.state_dict())
            for key,t in original.state_dict().items():assert torch.equal(t,candidate.state_dict()[key])
            # Isolate every layer before checking full dropout/backward/Adam.
            for layer,block in zip(original.layers,blocks):
                x=torch.randn(block.num_src_nodes(),8,device=device)
                close(layer(block,(x,x[:block.num_dst_nodes()])),functional_layer(layer,block,x),result)
                result['layer_cases']+=1
            optimizers=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=.001) for m in (original,candidate)]
            for step in range(3):
                data=torch.randn(8,8,device=device);labels=torch.tensor([0,2,5],device=device)
                state=rng_state(device);predictions=[];gradients=[];input_grads=[];end_rng=[]
                for m,opt in zip((original,candidate),optimizers):
                    restore_rng(state,device);m.train();x=data.clone().requires_grad_(True)
                    opt.zero_grad(set_to_none=True)
                    pred=m(blocks,x);loss=torch.nn.functional.cross_entropy(pred,labels)
                    loss.backward();predictions.append(pred.detach());input_grads.append(x.grad)
                    gradients.append([p.grad.clone() if p.grad is not None else None for p in m.parameters()])
                    opt.step();end_rng.append(rng_state(device))
                close(*predictions,result);close(*input_grads,result)
                for a,b in zip(*gradients):close(a,b,result)
                for a,b in zip(original.parameters(),candidate.parameters()):close(a,b,result)
                sa,sb=[o.state_dict() for o in optimizers]
                assert sa['param_groups']==sb['param_groups']
                for key in sa['state']:
                    for item in sa['state'][key]:close(sa['state'][key][item],sb['state'][key][item],result)
                for a,b in zip(*end_rng):
                    if a is not None:assert torch.equal(a,b)
                result['model_updates']+=2
            original.eval();candidate.eval()
            x=torch.randn(8,8,device=device)
            close(original(blocks,x),candidate(blocks,x),result)
    return result
