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
    a=SAGE(128,128,172,3,.2).to(device);b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001,weight_decay=0.);ob=torch.optim.Adam(b.parameters(),lr=.001,weight_decay=0.)
    initial={k:v.detach().clone() for k,v in b.state_dict().items()}
    optimize(b,ob,capture=capture,batch_size=4)
    assert set(initial)==set(b.state_dict())
    for k,v in initial.items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    blocks=fixture_blocks(device);x=torch.randn(24,128,device=device);target=torch.arange(4,device=device)%172
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
    return dict(passed=True,frontier_cases=frontier_check('cuda'),model=model_check('cuda',True),pair=pair_check(native),ssd_access=False)


def pair_check(m):
    import numpy as np
    cases=0
    for seed in range(12):
        rng=np.random.RandomState(seed);ids=rng.permutation(2048)[:1300].astype(np.int64)
        flags=rng.randint(0,2,len(ids)).astype(np.bool_);cpu=(rng.random_sample(2048)<.25).astype(np.int32)
        eligible={int(row):i for i,row in enumerate(ids) if flags[i] and row%8<2 and not cpu[row]}
        wanted=np.array([eligible.get(int(row)^1,-1) if int(row) in eligible else -1 for row in ids],np.int32)
        ts=[torch.from_numpy(x).cuda() for x in (ids,flags,cpu)];cap=1<<(2*len(ids)-1).bit_length()
        table=torch.empty(cap,dtype=torch.int32,device='cuda');partners=torch.empty(len(ids),dtype=torch.int32,device='cuda');errors=torch.zeros(1,dtype=torch.int64,device='cuda')
        m.probe_pair_hash(*[t.data_ptr() for t in ts],len(ids),len(cpu),table.data_ptr(),cap,partners.data_ptr(),errors.data_ptr())
        assert errors.item()==0 and np.array_equal(partners.cpu().numpy(),wanted)
        cases+=1
    src=torch.arange(256,dtype=torch.float32,device='cuda');dest=torch.full((2048,),-77.,device='cuda')
    a=dest[128:256];b=dest[1024:1152]
    m.probe_pair_scatter(src.data_ptr(),a.data_ptr(),b.data_ptr())
    expected=torch.full_like(dest,-77.);expected[128:256]=src[:128];expected[1024:1152]=src[128:]
    assert torch.equal(dest,expected)
    return dict(passed=True,cases=cases,row_bytes=512,slot_stride_rows=8,scatter_bit_exact=True)
