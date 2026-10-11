"""Functional mean-SAGE preserving each UKS layer's original operator order."""
import torch
from torch import nn
import torch.nn.functional as F
import dgl.ops
from ae.igb.models import SAGE

class FinalSelfAdd(nn.Module):
    def __init__(self,layer):
        super().__init__();self.fc_self=layer.fc_self
    def forward(self,destination,neighbor):return self.fc_self(destination)+neighbor

def neighbor(layer,block,h):
    # SAGEConv transforms first precisely when input width exceeds output width.
    if layer._in_src_feats>layer._out_feats:
        return dgl.ops.copy_u_mean(block,layer.fc_neigh(h))
    return layer.fc_neigh(dgl.ops.copy_u_mean(block,h))

class FastSAGE(SAGE):
    def forward(self,blocks,x):
        h=x
        for i,(layer,block) in enumerate(zip(self.layers,blocks)):
            dst=h[:block.num_dst_nodes()];neigh=neighbor(layer,block,h)
            if i==len(self.layers)-1 and self.training and torch.is_grad_enabled() and getattr(self,'_capture',None) is not None and len(dst)==self._capture_batch:
                h=self._capture(dst,neigh);self._capture_calls+=1
            else:h=layer.fc_self(dst)+neigh
            if i!=len(self.layers)-1:h=F.relu(self.dropout(h))
        return h

def optimize(net,opt,capture=False,batch_size=1024):
    for layer in net.layers:
        if layer._aggre_type!='mean' or layer.feat_drop.p!=0 or layer.activation is not None or layer.norm is not None:
            raise ValueError('Unsupported SAGE layer')
    params=tuple(net.parameters());ids=[id(p) for p in params]
    if ids!=[id(p) for g in opt.param_groups for p in g['params']] or opt.state:raise RuntimeError('Untouched optimizer required')
    net.__class__=FastSAGE;net._capture_calls=0
    if capture:
        if not net.training or params[0].device.type!='cuda' or any(p.dtype!=torch.float32 or p.grad is not None for p in params):raise RuntimeError('Capture requires untouched FP32 CUDA model')
        device=params[0].device
        saved=[p.detach().clone() for p in params];rng=torch.get_rng_state();crng=torch.cuda.get_rng_state(device)
        final=net.layers[-1]
        if final._in_src_feats<=final._out_feats:raise RuntimeError('UKS final layer must transform before aggregate')
        args=(torch.zeros(batch_size,final._in_dst_feats,device=device,requires_grad=True),torch.zeros(batch_size,final._out_feats,device=device,requires_grad=True))
        wrapper=torch.cuda.make_graphed_callables(FinalSelfAdd(final),args,num_warmup_iters=3)
        if not all(torch.equal(a,b) and b.grad is None for a,b in zip(saved,params)):raise RuntimeError('Capture changed parameters')
        if not torch.equal(rng,torch.get_rng_state()) or not torch.equal(crng,torch.cuda.get_rng_state(device)):raise RuntimeError('Capture consumed RNG')
        object.__setattr__(net,'_capture',wrapper);net._capture_batch=batch_size
    if ids!=[id(p) for p in net.parameters()] or opt.state:raise RuntimeError('Parameter/optimizer identity changed')
    return net
