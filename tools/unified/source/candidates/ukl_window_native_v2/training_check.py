"""Tiny three-layer SAGE/Adam validation of local blocks from native compact arrays."""
import hashlib,json
from pathlib import Path
import numpy as np
from .tests import fixture
from .sampler import Sampler

def main():
 import torch,dgl
 from dgl.nn import SAGEConv
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert not torch.cuda.is_initialized()
 g=fixture(True);x=torch.arange(67*128,dtype=torch.float32).reshape(67,128)/10000;labels=torch.arange(67)%19
 def run(grouped,reference):
  torch.manual_seed(23);sampler=Sampler(g,grouped=grouped,python_reference=reference)
  layers=torch.nn.ModuleList([SAGEConv(128,128,'mean'),SAGEConv(128,128,'mean'),SAGEConv(128,19,'mean')]);optimizer=torch.optim.Adam(layers.parameters(),lr=.001,weight_decay=.001)
  losses=[];trace=hashlib.sha256()
  for roots in ([0,1,2,3],[5,16,17,66],[0,2,5,7]):
   inp,out,blocks=sampler.sample_blocks(roots);h=x[inp]
   for block in blocks:
    for t in [block.srcdata[dgl.NID],block.dstdata[dgl.NID],block.edata[dgl.EID]]:trace.update(t.numpy().tobytes())
   for i,(layer,block) in enumerate(zip(layers,blocks)):
    h=layer(block,h)
    if i<2:h=torch.nn.functional.dropout(torch.relu(h),p=.2,training=True)
   loss=torch.nn.functional.cross_entropy(h,labels[out]);assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
  state=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in layers.state_dict().values())).hexdigest()
  return dict(losses=losses,model_sha256=state,sample_trace_sha256=trace.hexdigest(),updates=3)
 results={}
 for grouped in (False,True):
  actual=run(grouped,False);expected=run(grouped,True);assert actual==expected
  results['digit' if grouped else 'gids']=actual
 value=dict(passed=True,per_arm_native_python_trace_loss_model_exact=True,results=results,cuda_initialized=torch.cuda.is_initialized(),real_features_or_ssd=False,full_graph_dgl=False)
 p=Path(__file__).resolve().parents[2]/'results/ukl_window_native_20260930_v2/checks/training.json';assert not p.exists();p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
if __name__=='__main__':main()
