"""Owned anonymous registration; bounded calls and unconditional cleanup.

Small mode is capped at 256 MiB. Full mode requires the exact UKL budget.
The preload callback must finish/synchronize each bounded batch before return.
"""
import ctypes as C
import mmap
from candidates.ukl_real_multi_v9.memory import assert_anonymous

MIB=2**20
MAX_BYTES=256*MIB
CHUNK=16*MIB
RETAINED=[]

class Cancelled(RuntimeError):pass

class SegmentedArena:
    def __init__(self,size,api,full=False,interleave=False):
        limit=((78780147*512+4095)//4096*4096) if full else MAX_BYTES
        if type(size) is not int or not 0<size<=limit or size%4096:
            raise ValueError('Page-aligned allocation within explicit UKL cache budget required')
        if full and size!=limit:raise ValueError('Full mode requires exact UKL cache size')
        self.size,self.api=size,api
        self.mm=mmap.mmap(-1,size,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,
                          prot=mmap.PROT_READ|mmap.PROT_WRITE)
        self.address=C.addressof(C.c_char.from_buffer(self.mm))
        self.registered=[];self.device_address=None;self.closed=False
        try:
            self.mm.madvise(mmap.MADV_DONTFORK)
            assert_anonymous(self.address,self.size)
            if interleave:
                from .numa import bind_fresh
                bind_fresh(self,'cpu_features')
        except BaseException:
            self.mm.close();self.closed=True
            raise

    def prepare(self,check,event=lambda **k:None):
        if self.closed or self.registered:raise RuntimeError('Fresh owned allocation required')
        for off in range(0,self.size,CHUNK):
            check();n=min(CHUNK,self.size-off)
            C.memset(self.address+off,0,n)
            check();assert_anonymous(self.address+off,n)
            self.api.register(self.address+off,n)
            # Track immediately, including when device-pointer lookup fails.
            self.registered.append((off,n))
            dev=self.api.device_pointer(self.address+off)
            if self.device_address is None:self.device_address=dev-off
            if dev!=self.device_address+off:raise RuntimeError('Noncontiguous GPU aliases')
            event(stage='allocation_chunk',chunks=len(self.registered),bytes=off+n)
            check()

    def preload(self,callback,check,event=lambda **k:None):
        if self.closed or sum(n for _,n in self.registered)!=self.size:
            raise RuntimeError('Fully registered allocation required')
        for index,(off,n) in enumerate(self.registered):
            check()
            callback(self.address+off,self.device_address+off,n,index)
            event(stage='preload_chunk',chunks=index+1,bytes=off+n)
            check()

    def close(self,event=lambda **k:None):
        if self.closed:return
        try:
            # Cancellation does not interrupt cleanup or drop ownership early.
            self.api.synchronize()
            while self.registered:
                off,n=self.registered[-1]
                self.api.unregister(self.address+off)
                self.registered.pop()
                event(stage='release_chunk',remaining=len(self.registered))
            self.mm.close();self.mm=None;self.closed=True
        except BaseException:
            RETAINED.append(self)
            raise

    def receipt(self):
        return dict(size=self.size,registered_remaining=len(self.registered),
                    released=self.closed,anonymous_private=True,chunk_bytes=CHUNK)
