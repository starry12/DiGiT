"""Local-block adapter for the bounded native prototype, not a full-graph loader."""
import numpy as np
from .native import sample
from .reference import sample as reference
class Sampler:
 def __init__(self,graph,grouped=False,seed=0,gpu=False,python_reference=False):
  self.g=graph;self.grouped=grouped;self.seed=seed;self.gpu=gpu;self.reference=python_reference
 def sample_blocks(self,targets):
  import torch,dgl
  from candidates.ukl_sage_compact_v1.sampler import block_from_edges,STORAGE_ROW
  seeds=np.asarray(targets,dtype=np.int32);output=seeds.copy();blocks=[]
  for layer in (2,1,0):
   f=[10,5,5][layer];grouped=self.grouped and layer==0
   if self.reference:rows=reference(self.g,seeds,f,grouped,self.seed+layer)
   else:
    raw=sample(self.g,seeds,f,grouped,self.seed+layer,self.gpu)
    rows=[{k:raw[k][i*f:i*f+int(c)].tolist() for k in ('src','dst','eid','rows')} for i,c in enumerate(raw['counts'])]
   src=[x for r in rows for x in r['src']];dst=[x for r in rows for x in r['dst']];eids=[x for r in rows for x in r['eid']]
   block=block_from_edges(seeds,src,dst,eids);seeds=block.srcdata[dgl.NID].numpy().astype(np.int32)
   if layer==0:
    preferred={}
    for r in rows:
     for node,address in zip(r['src'],r['rows']):preferred.setdefault(node,address)
    block.srcdata[STORAGE_ROW]=torch.tensor([preferred.get(int(n),int(self.g['primary'][n])) for n in seeds],dtype=torch.int64)
   blocks.insert(0,block)
  return blocks[0].srcdata[dgl.NID],torch.tensor(output.astype(np.int64)),blocks
