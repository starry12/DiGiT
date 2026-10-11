"""Map each original CSC array once and attach compact sparse metadata.
No graph-sized copy, full EID array or DGL graph. CUDA full-graph registration
is deliberately a separate admission/acceptance step, not fixture arena reuse.
"""
from pathlib import Path
import ctypes as C
import json
import numpy as np
from .native import BUILD,View,Out,view,outputs
FILES=dict(gptr=('group_ptr.i64',np.int64),gidx=('group_ids.i32',np.int32),covered=('covered.i64',np.int64),members=('members.i32',np.int32),bases=('bases.i64',np.int64),primary=('primary.i64',np.int64))

def array(path,dtype,count):
 path=Path(path)
 if path.stat().st_size!=count*np.dtype(dtype).itemsize:raise ValueError('Array extent changed '+str(path))
 return np.memmap(path,dtype=dtype,mode='r',shape=(count,)) if count else np.empty(0,dtype=dtype)

class CSC:
 def __init__(self,folder):
  folder=Path(folder);meta=json.loads((folder/'csc.json').read_text())
  if not meta.get('complete') or meta.get('orientation')!='incoming':raise ValueError('CSC not completed/incoming')
  self.folder=folder;self.nodes=n=meta['nodes'];self.edges=e=meta['edges']
  if not 0<n<2**31 or not 0<=e<2**63:raise ValueError('CSC extent')
  self.ptr=array(folder/'indptr.i64',np.int64,n+1);self.idx=array(folder/'indices.i32',np.int32,e)
  if self.ptr[0]!=0 or self.ptr[-1]!=e:raise ValueError('CSC endpoints')

class Overlay:
 def __init__(self,csc,folder):
  folder=Path(folder);meta=json.loads((folder/'overlay.json').read_text());n=csc.nodes;g=meta['groups']
  if not meta.get('complete') or meta['nodes']!=n or meta['edges']!=csc.edges or not 0<=g<2**31-n:raise ValueError('Overlay extent')
  self.csc=csc;self.arrays=dict(ptr=csc.ptr,idx=csc.idx)
  extents=dict(gptr=n+1,gidx=g,covered=2*g,members=2*g,bases=g,primary=n)
  for k,(name,dtype) in FILES.items():self.arrays[k]=array(folder/name,dtype,extents[k])
  self.arrays['members']=self.arrays['members'].reshape(g,2)
  gp=self.arrays['gptr']
  if gp[0]!=0 or gp[-1]!=g:raise ValueError('Overlay endpoints')
  # Evidence binding is checked by the build/launch controller. These cheap
  # structural checks are not a replacement for full preparation acceptance.
 def sample_cpu(self,seeds,fanout,grouped=True,seed=0):
  seeds=np.asarray(seeds)
  if seeds.dtype!=np.int32 or seeds.ndim!=1 or not 0<len(seeds)<=4096 or seeds.min()<0 or seeds.max()>=self.csc.nodes or type(fanout) is not int or not 1<=fanout<=32:raise ValueError('Sample bounds')
  seeds=np.ascontiguousarray(seeds);out=outputs(len(seeds),fanout);fn=C.CDLL(str(BUILD/'libcompact_cpu.so')).sample_cpu
  fn.argtypes=[View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,Out];fn.restype=C.c_int
  rc=fn(view(self.arrays),int(seeds.ctypes.data),len(seeds),fanout,int(grouped),seed,Out(*[int(v.ctypes.data) for v in out.values()]))
  if rc or np.any(out['errors']):raise ValueError('Native sample rejected input')
  return out
