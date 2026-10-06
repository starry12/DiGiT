"""Bounded ctypes prototype. No full graph, SSD, features or training allocation."""
import ctypes as C
import mmap
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BUILD=HERE.parents[1]/'results/ukl_sparse_overlay_20260930_v3/build'
class View(C.Structure):
 _fields_=[('nodes',C.c_int32)]+[(k,C.c_int64) for k in ('edges','origin','units','unit_origin','groups')]+[(k,C.c_void_p) for k in ('ptr','gptr','bases','primary','covered','idx','gidx','members')]
class Out(C.Structure):
 _fields_=[(k,C.c_void_p) for k in ('src','dst','counts','errors','eid','rows')]

def graph(ptr,idx,*,gptr=None,gidx=None,members=None,bases=None,primary=None,covered=None):
 from .layout import validate
 a=validate(dict(ptr=ptr,idx=idx,gptr=gptr,gidx=gidx,members=members,bases=bases,primary=primary,covered=covered))
 if len(ptr)>4097 or len(idx)>131072:raise ValueError('Native prototype is fixture-only')
 return a

def view(a,pointers=None):
 p=pointers or {k:int(v.ctypes.data) for k,v in a.items()}
 return View(len(a['ptr'])-1,len(a['idx']),int(a['ptr'][0]),len(a['gidx']),int(a['gptr'][0]),len(a['members']),*[p[k] for k in ('ptr','gptr','bases','primary','covered','idx','gidx','members')])

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
