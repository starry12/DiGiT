"""Bounded ctypes prototype. No full graph, SSD, features or training allocation."""
import ctypes as C
import mmap
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BUILD=HERE.parents[1]/'results/ukl_window_native_20260930_v2/build'
class View(C.Structure):
 _fields_=[('nodes',C.c_int32)]+[(k,C.c_int64) for k in ('edges','origin','units','unit_origin','groups')]+[(k,C.c_void_p) for k in ('ptr','gptr','bases','primary','idx','gidx','members')]
class Out(C.Structure):
 _fields_=[(k,C.c_void_p) for k in ('src','dst','counts','errors','eid','rows')]

def graph(ptr,idx,*,gptr=None,gidx=None,members=None,bases=None,primary=None):
 arrays=dict(ptr=ptr,idx=idx,gptr=gptr,gidx=gidx,members=members,bases=bases,primary=primary)
 n=len(ptr)-1
 if not 0<n<=4096 or len(idx)>131072:raise ValueError('Native prototype is fixture-only')
 for name,dtype in [('ptr',np.int64),('idx',np.int32)]:
  if arrays[name].dtype!=dtype or arrays[name].ndim!=1:raise ValueError('Wrong dtype/rank '+name)
 if ptr[0]<0 or ptr[-1]-ptr[0]!=len(idx) or np.any(ptr[1:]<ptr[:-1]):raise ValueError('Bad original offsets')
 if len(idx) and (idx.min()<0 or idx.max()>=n):raise ValueError('Bad original nodes')
 if primary is None:arrays['primary']=np.arange(n,dtype=np.int64)
 if arrays['primary'].dtype!=np.int64 or arrays['primary'].shape!=(n,) or np.any(arrays['primary']<0) or np.any(arrays['primary']>=(2**63-1)//512):raise ValueError('Bad storage primary')
 if gptr is None:
  if any(x is not None for x in (gidx,members,bases)):raise ValueError('Partial group metadata')
  arrays.update(gptr=np.zeros(n+1,np.int64),gidx=np.empty(0,np.int32),members=np.empty((0,2),np.int32),bases=np.empty(0,np.int64))
 else:
  if gptr.dtype!=np.int64 or gptr.shape!=(n+1,) or gptr[0]<0 or gptr[-1]-gptr[0]!=len(gidx) or np.any(gptr[1:]<gptr[:-1]):raise ValueError('Bad group offsets')
  if gidx.dtype!=np.int32 or gidx.ndim!=1 or len(gidx)>131072:raise ValueError('Bad unit width')
  if members.dtype!=np.int32 or members.ndim!=2 or members.shape[1]!=2 or len(members)>65536:raise ValueError('Bad group members')
  if members.size and (members.min()<0 or members.max()>=n):raise ValueError('Bad member range')
  if bases.dtype!=np.int64 or bases.shape!=(len(members),) or np.any(bases<0) or np.any(bases>=(2**63-1)//512-1):raise ValueError('Bad group addresses')
  if len(gidx) and (gidx.min()<0 or gidx.max()>=n+len(members)):raise ValueError('Bad group IDs')
 return {k:np.ascontiguousarray(v) for k,v in arrays.items()}

def view(a,pointers=None):
 p=pointers or {k:int(v.ctypes.data) for k,v in a.items()}
 return View(len(a['ptr'])-1,len(a['idx']),int(a['ptr'][0]),len(a['gidx']),int(a['gptr'][0]),len(a['members']),*[p[k] for k in ('ptr','gptr','bases','primary','idx','gidx','members')])

def outputs(count,f):
 return {k:np.full(count if k in ('counts','errors') else count*f,-1,dtype=np.int64 if k in ('eid','rows') else np.int32) for k in ('src','dst','counts','errors','eid','rows')}

def sample(a,seeds,fanout,grouped=False,seed=0,gpu=False):
 # Revalidate the bounded Python contract before entering unchecked native code.
 a=graph(**a)
 if type(fanout) is not int or not 0<fanout<=32:raise ValueError('fanout outside prototype bound')
 if seeds.dtype!=np.int32 or seeds.ndim!=1 or not 0<len(seeds)<=4096 or seeds.min()<0 or seeds.max()>=len(a['ptr'])-1:raise ValueError('bad seeds')
 seeds=np.ascontiguousarray(seeds);out=outputs(len(seeds),fanout)
 lib=C.CDLL(str(BUILD/('libcompact_cuda.so' if gpu else 'libcompact_cpu.so')));fn=lib.sample_cuda if gpu else lib.sample_cpu
 fn.argtypes=[View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,Out];fn.restype=C.c_int
 if not gpu:
  code=fn(view(a),int(seeds.ctypes.data),len(seeds),fanout,int(grouped),seed,Out(*[int(v.ctypes.data) for v in out.values()]))
 else:code=cuda_call(fn,a,seeds,fanout,grouped,seed,out)
 if code:raise RuntimeError('Native API status '+str(code))
 if np.any(out['errors']):raise ValueError('Native sample rejected graph '+str(out['errors'].tolist()))
 return out

def cuda_call(fn,a,seeds,f,grouped,seed,out):
 rt=C.CDLL('/usr/local/cuda/lib64/libcudart.so')
 rt.cudaHostRegister.argtypes=[C.c_void_p,C.c_size_t,C.c_uint];rt.cudaHostGetDevicePointer.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint]
 rt.cudaHostUnregister.argtypes=[C.c_void_p];rt.cudaMalloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t]
 rt.cudaFree.argtypes=[C.c_void_p];rt.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int];rt.cudaMemset.argtypes=[C.c_void_p,C.c_int,C.c_size_t]
 def ok(rc):
  if rc:raise RuntimeError('CUDA runtime status '+str(rc))
 # One page-aligned host arena keeps tiny NumPy arrays from registering the same page twice.
 # This copy is fixture-only; production needs direct registration of readonly mmap ranges.
 offsets={};size=0
 for k,v in a.items():offsets[k]=size;size+=max(4096,((v.nbytes+4095)//4096)*4096)
 arena=mmap.mmap(-1,size);buffer=(C.c_char*size).from_buffer(arena);address=C.addressof(buffer);registered=False;allocated=[]
 try:
  for k,v in a.items():
   if v.nbytes:C.memmove(address+offsets[k],int(v.ctypes.data),v.nbytes)
  ok(rt.cudaHostRegister(address,size,2));registered=True;device=C.c_void_p();ok(rt.cudaHostGetDevicePointer(C.byref(device),address,0))
  def alloc(n):
   p=C.c_void_p();ok(rt.cudaMalloc(C.byref(p),n));allocated.append(p);return p
  seeds_gpu=alloc(seeds.nbytes);ok(rt.cudaMemcpy(seeds_gpu,int(seeds.ctypes.data),seeds.nbytes,1))
  ptr={}
  for k,v in out.items():ptr[k]=alloc(v.nbytes);ok(rt.cudaMemset(ptr[k],255,v.nbytes))
  code=fn(view(a,{k:device.value+off for k,off in offsets.items()}),seeds_gpu,len(seeds),f,int(grouped),seed,Out(*[p.value for p in ptr.values()]))
  for k,v in out.items():ok(rt.cudaMemcpy(int(v.ctypes.data),ptr[k],v.nbytes,2))
  return code
 finally:
  for p in reversed(allocated):rt.cudaFree(p)
  if registered:rt.cudaHostUnregister(address)
  del buffer;arena.close()
