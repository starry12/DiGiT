"""Small semantic fixtures; GPU mode does not open SSD or load UKS."""
import copy
import torch,dgl
from ae.igb.models import SAGE
from .model import optimize
from candidates.pa_sage_512b_latency_v1.frontier import coo_frontier

def frontier_check(device):
    cases=0
    for src,dst,n in [([],[],4),([0,1,1,3],[1,1,1,0],5),([0,1,2],[0,1,2],4)]:
        a=torch.tensor(src,device=device,dtype=torch.int64);b=torch.tensor(dst,device=device,dtype=torch.int64)
        original=dgl.graph((a,b),num_nodes=n);fast=coo_frontier(a,b,n,check_bounds=True)
        original.edata[dgl.EID]=torch.arange(len(a),device=device);fast.edata[dgl.EID]=original.edata[dgl.EID]
        seeds=torch.tensor([0,1,2],device=device)
        x,y=dgl.to_block(original,seeds),dgl.to_block(fast,seeds)
        for u,v in [(x.srcdata[dgl.NID],y.srcdata[dgl.NID]),(x.dstdata[dgl.NID],y.dstdata[dgl.NID]),(x.edata[dgl.EID],y.edata[dgl.EID]),*zip(x.edges(),y.edges())]:
            torch.testing.assert_close(u,v,rtol=0,atol=0)
        cases+=1
    return cases

def fixture_blocks(device):
    blocks=[]
    for ns,nd in [(24,16),(16,8),(8,4)]:
        src=torch.tensor([i%ns for i in range(nd*3)],device=device)
        # Last destination is isolated, and other rows include duplicates.
        dst=torch.tensor([i%(nd-1) for i in range(nd*3)],device=device)
        blocks.append(dgl.create_block((src,dst),num_src_nodes=ns,num_dst_nodes=nd))
    return blocks

def model_check(device,capture):
    torch.set_num_threads(1);torch.manual_seed(31)
    a=SAGE(256,128,19,3,.2).to(device);b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001,weight_decay=.001);ob=torch.optim.Adam(b.parameters(),lr=.001,weight_decay=.001)
    initial={k:v.detach().clone() for k,v in b.state_dict().items()}
    optimize(b,ob,capture=capture,batch_size=4)
    assert set(initial)==set(b.state_dict())
    for k,v in initial.items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    blocks=fixture_blocks(device);x=torch.randn(24,256,device=device);target=torch.arange(4,device=device)%19
    maximum=0.
    for step in range(8):
        left=x.detach().clone().requires_grad_();right=x.detach().clone().requires_grad_()
        torch.manual_seed(41+step);pa=a(blocks,left);la=torch.nn.functional.cross_entropy(pa,target);oa.zero_grad(set_to_none=True);la.backward()
        torch.manual_seed(41+step);pb=b(blocks,right);lb=torch.nn.functional.cross_entropy(pb,target);ob.zero_grad(set_to_none=True);lb.backward()
        maximum=max(maximum,float((pa-pb).abs().max()))
        torch.testing.assert_close(pa,pb,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-6)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
        oa.step();ob.step()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
            for name in oa.state[p]:torch.testing.assert_close(oa.state[p][name],ob.state[q][name],rtol=1e-5,atol=1e-6)
    assert not capture or b._capture_calls==8
    return dict(passed=True,updates=8,max_logit_difference=maximum,capture=capture,capture_calls=b._capture_calls)

def gpu_check():
    from .runtime import prepare_imports
    cls,native=prepare_imports()
    assert native.uks_async_policy_api()==1
    fs=native.BAM_Feature_Store_float()
    try:fs.prepare_policy_async(0,0);raise AssertionError('Invalid admission accepted')
    except RuntimeError:pass
    return dict(passed=True,frontier_cases=frontier_check('cuda'),model=model_check('cuda',True),ssd_access=False)
