"""The only independent variable is explicit reverse-edge augmentation."""
import gc
import numpy as np
import torch,dgl

def build(edges,n,arm,notify=lambda **kw:None):
 if arm not in ('directed','bidirectional'):raise ValueError(arm)
 # Data audit establishes no source self edges; enforce again during the copy.
 e=len(edges);total=e*(2 if arm=='bidirectional' else 1)+n
 src=torch.empty(total,dtype=torch.int64);dst=torch.empty_like(src)
 for lo in range(0,e,8_000_000):
  hi=min(e,lo+8_000_000);u=np.asarray(edges[lo:hi,0]);v=np.asarray(edges[lo:hi,1])
  if (u==v).any():raise ValueError('Source self edges require a new explicit normalization contract')
  src[lo:hi].copy_(torch.from_numpy(u.copy()));dst[lo:hi].copy_(torch.from_numpy(v.copy()))
  if arm=='bidirectional':src[e+lo:e+hi].copy_(torch.from_numpy(v.copy()));dst[e+lo:e+hi].copy_(torch.from_numpy(u.copy()))
  if lo%80_000_000==0:notify(stage='graph_coo_copy',done=hi,total=e)
 base=total-n
 for lo in range(0,n,8_000_000):
  hi=min(n,lo+8_000_000);ids=torch.arange(lo,hi,dtype=torch.int64);src[base+lo:base+hi]=ids;dst[base+lo:base+hi]=ids
 notify(stage='graph_convert_csc',edges=total)
 g=dgl.graph((src,dst),num_nodes=n,idtype=torch.int64).formats('csc');g.create_formats_()
 del src,dst;gc.collect()
 assert g.num_nodes()==n and g.num_edges()==total and g.formats()['created']==['csc']
 return g

def verify_sample(blocks,edges,n,arm):
 e=len(edges);base=e*(2 if arm=='bidirectional' else 1)
 for b in blocks:
  u,v=b.edges(order='eid');src=b.srcdata[dgl.NID][u].cpu().numpy();dst=b.dstdata[dgl.NID][v].cpu().numpy();ids=b.edata[dgl.EID].cpu().numpy()
  assert (ids>=0).all() and (ids<base+n).all()
  forward=ids<e;reverse=(ids>=e)&(ids<base);selfs=ids>=base
  assert np.array_equal(src[forward],edges[ids[forward],0]) and np.array_equal(dst[forward],edges[ids[forward],1])
  assert np.array_equal(src[reverse],edges[ids[reverse]-e,1]) and np.array_equal(dst[reverse],edges[ids[reverse]-e,0])
  assert np.array_equal(src[selfs],dst[selfs]) and np.array_equal(src[selfs],ids[selfs]-base)
