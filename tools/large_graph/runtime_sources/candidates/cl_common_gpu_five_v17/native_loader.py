"""Bounded CPU pointers and private native ABI; never register mapped files."""
import ctypes as C
import hashlib,json
import numpy as np
from .sampler_build import OUT,ROOT
from candidates.ukl_native_sampling_v10 import sampling as S

def library(name):
    p=OUT/name;r=json.loads((OUT/(name+'.json')).read_text())
    if not r['passed'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('Optimized sampler binary changed')
    for rel,sha in r['sources'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=sha:raise RuntimeError('Optimized sampler source changed')
    return C.CDLL(str(p))

def load_sampler(gpu):
    lib=library('libcompact_cuda.so' if gpu else 'libcompact_cpu.so')
    fn=lib.sample_cuda if gpu else lib.sample_cpu
    fn.argtypes=[S.View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,S.Out];fn.restype=C.c_int
    return lib,fn

def load_validator():
    lib=library('libvalidate.so');fn=lib.validate_group_rows
    fn.argtypes=[S.View,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p];fn.restype=C.c_int
    return lib,fn

def validate_rows(fn,graph,seeds,fields,counts):
    src,rows=fields['src'],fields['rows']
    if (seeds.dtype!=np.int32 or counts.dtype!=np.int32 or src.dtype!=np.int32 or rows.dtype!=np.int64
        or seeds.ndim!=1 or counts.shape!=seeds.shape or src.ndim!=2 or rows.shape!=src.shape
        or src.shape[0]!=len(seeds) or not 0<src.shape[1]<=32 or not 0<src.size<=65536):
        raise ValueError('Bounded validation arrays required')
    arrays=[np.ascontiguousarray(x) for x in (seeds,src,rows,counts)]
    v=S.View(graph.nodes,graph.edges,graph.origin,graph.units,graph.unit_origin,graph.groups,
             *[int(graph.arrays[k].ctypes.data) for k in S._ORDER])
    rc=fn(v,arrays[0].ctypes.data,len(seeds),src.shape[1],*[x.ctypes.data for x in arrays[1:]])
    if rc:raise RuntimeError('Compiled group validation rejected output: '+str(rc))
