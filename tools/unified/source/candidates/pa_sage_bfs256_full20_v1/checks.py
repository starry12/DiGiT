"""Regressions for a captured full batch followed by an uncaptured epoch tail."""
from .common import setup,require

def tail_model_check(device='cpu',capture=False):
    setup()
    import copy
    import torch,dgl
    from ae.igb.models import SAGE
    from candidates.pa_sage_uks_style_v1.model import optimize
    torch.set_num_threads(1);torch.manual_seed(17)
    a=SAGE(128,128,172,3,.2).to(device);b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
    optimize(b,ob,capture=capture,batch_size=4)
    for step,batch in enumerate([4,3,4,3]):
        blocks=[]
        for ns,nd in [(24,16),(16,8),(8,batch)]:
            dst=torch.arange(nd,device=device).repeat_interleave(2);src=torch.arange(2*nd,device=device)%ns
            blocks.append(dgl.create_block((src,dst),num_src_nodes=ns,num_dst_nodes=nd))
        x=torch.randn(24,128,device=device);target=torch.arange(batch,device=device)
        for model,opt in [(a,oa),(b,ob)]:
            torch.manual_seed(30+step);pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,target)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)
            for key in oa.state[p]:torch.testing.assert_close(oa.state[p][key],ob.state[q][key],rtol=2e-5,atol=2e-6)
    require(b._capture_calls==(2 if capture else 0),'Tail incorrectly used fixed-size capture')
    return dict(passed=True,updates=4,batch_sizes=[4,3,4,3],capture_calls=b._capture_calls)

def gpu_check():
    from candidates.pa_sage_uks_style_v1.checks import gpu_check as parent
    result=parent();result['full_batch_tail_full_batch']=tail_model_check('cuda',True)
    return result
