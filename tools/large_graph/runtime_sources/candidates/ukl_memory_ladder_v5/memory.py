"""Private anonymous ownership; buffered file reads never registered with CUDA."""
import ctypes as C
import hashlib
import mmap
import os
from pathlib import Path
import numpy as np

CHUNK = 1024 * 1024
MAX_BYTES = 96 * 1024 * 1024 * 1024  # Independent stage-five ceiling; v6 unchanged.
RESERVE = 64 * 1024**3
RETAINED = []  # Retain registered mappings if synchronization/unregistration fails.

def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

def admission(size):
    if not 0 < size <= MAX_BYTES:
        raise RuntimeError('stage-five ceiling is 96 GiB; full graph disabled')
    m = dict((x.split(':')[0], int(x.split()[1])*1024) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:', 'Dirty:')))
    if m['MemAvailable'] < 2*size + CHUNK + RESERVE or m['Dirty'] > 2*1024**3:
        raise RuntimeError('Host memory/dirty-page admission refused')

def assert_anonymous(address, size):
    end = address + size
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(); lo, hi = (int(x,16) for x in fields[0].split('-'))
        if lo <= address < hi:
            if hi < end or fields[1] != 'rw-p' or fields[3] != '00:00' or fields[4] != '0' or len(fields)>5:
                raise RuntimeError('Only private anonymous writable VMAs may be registered')
            return
    raise RuntimeError('Anonymous allocation not found in process maps')

class Arena:
    """Own one private anonymous VMA. No external pointer adoption API."""
    def __init__(self, specs):
        # specs: key -> (path, dtype, shape, SHA256 of raw array file)
        self.specs = {}; self.offsets = {}; self.size = 0
        self.arrays = {}; self.mm = None; self.address = None; self.registered = False
        self.loaded = False; self.closed = False
        for name, (path, dtype, shape, digest) in specs.items():
            dt=np.dtype(dtype)
            if dt not in (np.dtype('int32'),np.dtype('int64')) or any(type(n)!=int or n<0 for n in shape):
                raise ValueError('Only integer graph arrays accepted')
            count=1
            for n in shape:count*=n
            size=count*dt.itemsize
            if len(digest)!=64:raise ValueError('Expected content SHA256 required')
            self.offsets[name]=self.size
            self.size += max(4096, (size+4095)//4096*4096)
            self.specs[name]=(Path(path),dt,tuple(shape),digest,size)
        admission(self.size)  # Before opening sources or allocating anything.
        self.mm=mmap.mmap(-1,self.size,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,prot=mmap.PROT_READ|mmap.PROT_WRITE)
        self.address=C.addressof(C.c_char.from_buffer(self.mm))
        assert_anonymous(self.address,self.size)

    def load(self, progress=None):
        if self.closed or self.loaded or self.registered:raise RuntimeError('Invalid load lifecycle')
        try:
            for name,(path,dt,shape,want,nbytes) in self.specs.items():
                admission(self.size)
                with path.open('rb',buffering=0) as f:
                    before=identity(os.fstat(f.fileno()))
                    if before[2]!=nbytes:raise ValueError('Source size mismatch: '+str(path))
                    digest=hashlib.sha256();off=0
                    while off<nbytes:
                        admission(self.size)
                        # readinto copies file bytes to anonymous pages; no file mmap.
                        target=memoryview(self.mm)[self.offsets[name]+off:self.offsets[name]+min(off+CHUNK,nbytes)]
                        try:
                            got=f.readinto(target)
                            if not got:raise IOError('Source truncated')
                            digest.update(target[:got])
                        finally:target.release()
                        off+=got
                        if progress:progress(name,off,nbytes)
                    if identity(os.fstat(f.fileno()))!=before or identity(path.stat())!=before:
                        raise RuntimeError('Source changed during load')
                    if digest.hexdigest()!=want:raise ValueError('Source digest mismatch: '+str(path))
                self.arrays[name]=np.frombuffer(self.mm,dtype=dt,count=nbytes//dt.itemsize,offset=self.offsets[name]).reshape(shape)
                self.arrays[name].flags.writeable=False
            self.loaded=True
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:return
        if self.registered:raise RuntimeError('Cannot release memory still registered with CUDA')
        self.arrays.clear()
        if self.mm is not None:self.mm.close()
        self.closed=True

class CudaAPI:
    def __init__(self):
        self.rt=C.CDLL('/usr/local/cuda/lib64/libcudart.so')
        for name,args in {'cudaHostRegister':[C.c_void_p,C.c_size_t,C.c_uint], 'cudaHostGetDevicePointer':[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint], 'cudaHostUnregister':[C.c_void_p], 'cudaDeviceSynchronize':[]}.items():
            fn=getattr(self.rt,name);fn.argtypes=args;fn.restype=C.c_int
    def call(self,name,*args):
        rc=getattr(self.rt,name)(*args)
        if rc:raise RuntimeError(name+' failed: '+str(rc))
    def register(self,address,size):self.call('cudaHostRegister',address,size,2)
    def device_pointer(self,address):
        p=C.c_void_p();self.call('cudaHostGetDevicePointer',C.byref(p),address,0);return p.value
    def synchronize(self):self.call('cudaDeviceSynchronize')
    def unregister(self,address):self.call('cudaHostUnregister',address)

class Registration:
    def __init__(self,arena,api=None):
        if type(arena) is not Arena or arena.closed or not arena.loaded or arena.registered:
            raise RuntimeError('Loaded owned Arena required')
        admission(arena.size);assert_anonymous(arena.address,arena.size)
        self.arena=arena;self.api=api if api is not None else CudaAPI();self.closed=False
        self.api.register(arena.address,arena.size);arena.registered=True
        RETAINED.append(self)  # Strong ownership until confirmed successful unregistration.
        try:
            base=self.api.device_pointer(arena.address)
            if not base:raise RuntimeError('Null CUDA device pointer')
            self.pointers={k:base+o for k,o in arena.offsets.items()}
        except BaseException:
            self.close();raise
    def close(self):
        if self.closed:return
        # Failure deliberately retains VMA and driver ownership, never unmaps underneath GPU.
        self.api.synchronize();self.api.unregister(self.arena.address)
        self.arena.registered=False;self.closed=True;RETAINED.remove(self)
