"""Small CUDA allocation/readback adapter, imported only by admitted worker."""
import ctypes as C
from candidates.ukl_real_multi_v9.memory import CudaAPI

class API(CudaAPI):
    def __init__(self):
        super().__init__()
        for name,args in {
            'cudaSetDevice':[C.c_int],'cudaFree':[C.c_void_p],
            'cudaMalloc':[C.POINTER(C.c_void_p),C.c_size_t],
            'cudaHostAlloc':[C.POINTER(C.c_void_p),C.c_size_t,C.c_uint],
            'cudaFreeHost':[C.c_void_p],
            'cudaMemset':[C.c_void_p,C.c_int,C.c_size_t],
            'cudaMemcpy':[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int],
        }.items():
            f=getattr(self.rt,name);f.argtypes=args;f.restype=C.c_int
        self.call('cudaSetDevice',0);self.call('cudaFree',None)
        self.scratch=None

    def roundtrip(self,host,device,size,index):
        if self.scratch is None:
            self.scratch=C.c_void_p();self.call('cudaMalloc',C.byref(self.scratch),16*2**20)
        pattern=(index%251)+1
        self.call('cudaMemset',self.scratch,pattern,size)
        self.call('cudaMemcpy',device,self.scratch,size,4)
        self.synchronize()
        # Entire chunk verified, not just the ends. Both allocations stay small.
        if C.string_at(host,size)!=bytes([pattern])*size:
            raise RuntimeError('GPU mapped-host roundtrip mismatch')

    def close(self):
        self.synchronize()
        if self.scratch is not None:
            self.call('cudaFree',self.scratch);self.scratch=None

    def reference_allocate(self,size):
        if size!=64*2**20:raise ValueError('Reference cudaHostAlloc fixed to 64 MiB')
        p=C.c_void_p();self.call('cudaHostAlloc',C.byref(p),size,2)
        return p

    def reference_release(self,p):
        self.synchronize();self.call('cudaFreeHost',p)
