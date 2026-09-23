"""Real forward/backward at maximum sampled-block row/edge envelopes; no SSD."""
import gc,time
from training.gat.common import *

def run_checks():
    setup()
    import torch,dgl,runner as r
    from training.gat.model import make_model
    from training.gat.admission import estimate
    from digit.gpu_admission import check_live
    r.startup();r.seed(0);p=cfg();plan=check_live(estimate())
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,9),'Expected one visible sm89 GPU')
    require(plan['passed'] and plan['free_bytes']>plan['total_bytes']-2**30,'GPU/host admission failed or selected GPU in use')
    bs=[];start=time.time();torch.cuda.reset_peak_memory_stats()
    for b,fanout in zip(plan['block_upper_envelope'],p['fanouts']):
        u=torch.arange(b['dst'],b['src'],device='cuda',dtype=torch.int64)
        v=torch.arange(b['dst'],device='cuda',dtype=torch.int64).repeat_interleave(fanout)
        require(len(u)==len(v)==b['edges'],'Wrong synthetic envelope')
        bs.append(dgl.create_block((u,v),num_src_nodes=b['src'],num_dst_nodes=b['dst']))
    model=make_model('cuda');options=dict(p['optimizer']['kwargs']);options['betas']=tuple(options['betas'])
    optimizer=torch.optim.Adam(model.parameters(),**options)
    x=torch.randn(bs[0].num_src_nodes(),128,device='cuda');labels=torch.arange(p['batch_size'],device='cuda')%172;losses=[]
    for step in range(3):
        pred=model(bs,x);loss=torch.nn.functional.cross_entropy(pred,labels);optimizer.zero_grad(set_to_none=True);loss.backward()
        require(all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters()),'Nonfinite maximum-envelope gradients')
        optimizer.step();torch.cuda.synchronize();losses.append(float(loss));del pred,loss
    train=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    model.eval();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        pred=model(bs,x);require(tuple(pred.shape)==(1024,172) and torch.isfinite(pred).all(),'Invalid maximum-envelope evaluation')
    torch.cuda.synchronize();evaluation=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    require(max(train['reserved'],evaluation['reserved'])<=plan['model_probe_limit_bytes'],'GAT model reserve underestimated')
    value=dict(passed=True,model='gat',model_parameter_count=p['model_parameter_count'],heads=4,hidden_per_head=128,hidden_concatenated=512,
               block_envelope=plan['block_upper_envelope'],training_updates=3,losses=losses,training_peak_bytes=train,evaluation_peak_bytes=evaluation,
               model_probe_limit_bytes=plan['model_probe_limit_bytes'],native_budget_bytes=plan['required_bytes'],seconds=time.time()-start,
               raw_ssd_access=False,scope='Synthetic maximum node/edge counts with disjoint neighbors and actual float32 GAT/Adam. Includes model, features and blocks only; full PA metadata/cache admission and native paired smoke still required.')
    del pred,model,optimizer,x,labels,bs;gc.collect();torch.cuda.empty_cache()
    return value
if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
