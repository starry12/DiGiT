"""Maximum fanout block allocation for each IG model, separate from native smoke."""
import gc
from candidates.ig_perf_v5.common import *
def run_checks():
    setup()
    import torch,dgl
    from candidates.ig_perf_v5.model import make_model,optimizer
    from candidates.ig_perf_v5.admission import check_live
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=cfg();plan=check_live();require(plan['passed'] and plan['free_bytes']>plan['total_bytes']-2**30,'Selected GPU occupied or budget failed')
    dimensions=[];dst=p['batch_size']
    for fanout in reversed(p['fanouts']):dimensions.insert(0,dict(src=dst*(fanout+1),dst=dst,edges=dst*fanout));dst*=fanout+1
    records={}
    for name in p['models']:
        torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();bs=[]
        for b,f in zip(dimensions,p['fanouts']):
            u=torch.arange(b['dst'],b['src'],device='cuda');v=torch.arange(b['dst'],device='cuda').repeat_interleave(f);bs.append(dgl.create_block((u,v),num_src_nodes=b['src'],num_dst_nodes=b['dst']))
        model=make_model(name,'cuda');opt=optimizer(model);x=torch.randn(bs[0].num_src_nodes(),1024,device='cuda');labels=torch.arange(p['batch_size'],device='cuda')%19;losses=[]
        for step in range(3):
            pred=model(bs,x);loss=torch.nn.functional.cross_entropy(pred,labels);opt.zero_grad(set_to_none=True);loss.backward();require(all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters()),'Nonfinite gradients');opt.step();losses.append(float(loss));del pred,loss
        train=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved());model.eval();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():pred=model(bs,x);require(tuple(pred.shape)==(1024,19) and torch.isfinite(pred).all(),'Bad evaluation')
        torch.cuda.synchronize();evaluation=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        require(max(train['reserved'],evaluation['reserved'])<=p['model_probe_limit_bytes'],'Model allocation exceeds reserve: '+name)
        records[name]=dict(losses=losses,train=train,validation=evaluation)
        del pred,model,opt,x,labels,bs,u,v;gc.collect();torch.cuda.empty_cache()
    return dict(passed=True,raw_ssd_access=False,models=records,block_envelope=dimensions,model_limit_bytes=p['model_probe_limit_bytes'],native_budget_bytes=plan['required_bytes'],scope='Synthetic maximum blocks/features/model only; native metadata/cache smoke still required')
if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
