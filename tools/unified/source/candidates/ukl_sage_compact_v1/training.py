"""Actual CPU SAGE/Adam on sampled local blocks, with no full graph conversion."""
import hashlib
import numpy as np
from .common import require,cpu_gate
from .data import epoch_order
from candidates.uks_sage_epoch_v1.epoch import Coverage


def epoch(p,sampler,selected,labels,fetch):
    import torch,dgl
    from ae.igb.models import SAGE
    from candidates.pa_sage_cache_policy_v1.training import seed_cpu,model_hash
    require(p.get('fixture') and p['feature_dim']==128 and p['epochs']==1 and p['evaluation']=='disabled','Wrong CPU protocol')
    cpu_gate();seed_cpu(p['seed'])
    model=SAGE(in_feats=128,h_feats=p['hidden'],num_classes=p['classes'],num_layers=3,dropout=p['dropout'])
    opt=torch.optim.Adam(model.parameters(),lr=p['optimizer']['lr'],weight_decay=p['optimizer']['weight_decay'])
    initial=model_hash(model);seed_cpu(p['seed']);model.train()
    roots,order=epoch_order(p,selected);coverage=Coverage(len(roots),p['batch_size']);trace=hashlib.sha256();losses=[]
    for lo in range(0,len(roots),p['batch_size']):
        cpu_gate();batch=roots[lo:lo+p['batch_size']];inp,out,blocks=sampler.sample_blocks(batch)
        require(np.array_equal(out.numpy(),batch),'Root order changed')
        x=fetch(inp,blocks);require(x.dtype==torch.float32 and x.shape==(len(inp),128) and torch.isfinite(x).all(),'Invalid features')
        logits=model(blocks,x);loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[batch].copy()))
        opt.zero_grad(set_to_none=True);loss.backward()
        require(torch.isfinite(logits).all() and torch.isfinite(loss) and
                all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()),'Nonfinite CPU training')
        opt.step();coverage.observe(len(batch));losses.append(float(loss.detach()))
        tensors=[inp,out,blocks[0].srcdata['digit_storage_row']]
        for b in blocks:
            u,v=b.edges(order='eid');tensors.extend((b.srcdata[dgl.NID],b.dstdata[dgl.NID],u,v,b.edata[dgl.EID]))
        for t in tensors:
            a=t.numpy();trace.update(str((a.shape,a.dtype.str)).encode());trace.update(a.tobytes())
    final=model_hash(model);require(initial!=final and all(torch.isfinite(v).all() for v in model.parameters()),'Bad final model')
    return dict(passed=True,fixture=True,native_execution=False,**coverage.finish(),initial_model_sha256=initial,
        final_model_sha256=final,root_sha256=order['root_sha256'],sample_trace_sha256=trace.hexdigest(),losses=losses,
        full_graph_dgl_constructed=False,full_eid_array_constructed=False,evaluation_calls=0,training_seconds=None)
