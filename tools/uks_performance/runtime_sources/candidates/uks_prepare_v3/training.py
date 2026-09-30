"""Actual tiny CPU GraphSAGE forward/backward/Adam with a replaceable feature source."""
import hashlib
import numpy as np
from .common import require,digest,cpu_gate
from .epoch import Coverage
from .data import batches,epoch_order
from .model import create


def cpu_epoch(p,sampler,graph,selected,labels,fetch):
    import torch,dgl
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    require(p.get('fixture') and graph.num_nodes()<=4096 and graph.num_edges()<=131072,'CPU fixture only')
    require(p['epochs']==1 and p['evaluation']=='disabled' and p['measurement']['warmup_batches']==0,'Wrong performance protocol')
    model,opt,initial=create(p);roots,order=epoch_order(p,selected);trace=hashlib.sha256();losses=[];inputs=0
    coverage=Coverage(len(roots),p['batch_size']);shapes=[]
    for batch in batches(roots,p['batch_size']):
        cpu_gate()
        inp,out,blocks=sampler.sample_blocks(graph,torch.from_numpy(batch.copy()))
        require(np.array_equal(out.numpy(),batch),'Sampler changed output roots')
        x=fetch(inp,blocks)
        require(x.shape==(len(inp),256) and x.dtype==torch.float32 and torch.isfinite(x).all(),'Bad 256D features')
        logits=model(blocks,x);loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[batch].copy()))
        opt.zero_grad(set_to_none=True);loss.backward()
        require(torch.isfinite(logits).all() and torch.isfinite(loss) and
                all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()),'Nonfinite training')
        opt.step();losses.append(float(loss.detach()));coverage.observe(len(out));inputs+=len(inp)
        tensors=[inp,out]
        for block in blocks:
            u,v=block.edges(order='eid');tensors.extend((block.srcdata[dgl.NID],block.dstdata[dgl.NID],u,v,block.edata[dgl.EID]))
        for t in tensors:
            a=t.numpy();trace.update(str((a.shape,a.dtype.str)).encode());trace.update(a.tobytes())
        shapes.append(dict(input_nodes=len(inp),output_nodes=len(out),edges=[b.num_edges() for b in blocks]))
    final=model_hash(model);require(initial!=final and all(torch.isfinite(v).all() for v in model.parameters()),'Model did not update correctly')
    require(not torch.cuda.is_initialized(),'CPU fixture started CUDA')
    return dict(passed=True,fixture=True,native_execution=False,feature_dim=256,evaluation_calls=0,warmup_batches=0,
        **coverage.finish(),initial_model_sha256=initial,final_model_sha256=final,root_sha256=order['root_sha256'],
        sample_trace_sha256=trace.hexdigest(),logical_requests=inputs,losses=losses,shapes=shapes,training_seconds=None)
