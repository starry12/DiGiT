"""Same RevPR/g2 algorithms with anonymous pointer input and bounded writes."""
import ctypes as C
import hashlib
import mmap
import os
from pathlib import Path
import time
import numpy as np
from . import protocol as P


class Native:
    CALLBACK=C.CFUNCTYPE(C.c_int,C.c_uint64,C.c_uint64,C.c_int)
    def __init__(self):
        self.rank=C.CDLL(str(P.OUT/'build/librevpr.so')).run_revpr
        self.group=C.CDLL(str(P.OUT/'build/libgroups.so')).run_groups
        self.rank.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64,C.c_uint64,C.c_char_p,C.c_int,self.CALLBACK,C.c_uint64]
        self.group.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64,C.c_uint64,C.c_void_p,C.c_uint64,C.c_char_p,C.c_uint64,self.CALLBACK,C.c_uint64]
        self.rank.restype=self.group.restype=C.c_int

    def run(self,kind,ptr,idx,out,hot=None,iterations=20,callback=lambda *a:None):
        if ptr.dtype!=np.int64 or idx.dtype!=np.int32 or not ptr.flags.c_contiguous or not idx.flags.c_contiguous:
            raise ValueError('Native pointer widths/contiguity')
        failure=[]
        def tick(*args):
            try:callback(*args);return 0
            except BaseException as e:failure.append(e);return 1
        cb=self.CALLBACK(tick);n=len(ptr)-1
        args=[ptr.ctypes.data,idx.ctypes.data,n,len(idx)]
        if kind=='rank':rc=self.rank(*args,os.fsencode(out),iterations,cb,P.APPLICATION_WRITE_RATE)
        elif kind=='group':
            if hot is None or hot.dtype!=np.int64 or not hot.flags.c_contiguous or len(hot)!=n//10:
                raise ValueError('Exact logical 10% hot set required')
            rc=self.group(*args,hot.ctypes.data,len(hot),os.fsencode(out),(n//5)//2,cb,P.APPLICATION_WRITE_RATE)
        else:raise ValueError('Unknown algorithm')
        if failure:raise failure[0]
        if rc:raise RuntimeError('Native CL '+kind+' failed: '+str(rc))


def direct_hash(path,check=lambda:None,rate=P.APPLICATION_READ_RATE):
    """Bounded actual disk readback of output, with no file-cache fallback."""
    path=Path(path);before=P.CL.ident(path.stat());size=before[2]
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW);buf=mmap.mmap(-1,8*P.MIB)
    h=hashlib.sha256();done=0;began=time.monotonic()
    try:
        while done<size:
            check();want=min(8*P.MIB,size-done);rounded=(want+4095)//4096*4096
            view=memoryview(buf)
            try:
                got=os.preadv(fd,[view[:rounded]],done)
                if got<want:raise IOError('Short direct output read')
                h.update(view[:want])
            finally:view.release()
            done+=want
            while done/rate>time.monotonic()-began:
                check();time.sleep(max(0,min(.1,done/rate-(time.monotonic()-began))))
        if P.CL.ident(os.fstat(fd))!=before or P.CL.ident(path.stat())!=before:
            raise RuntimeError('Output changed during hash')
    finally:buf.close();os.close(fd)
    return dict(path=str(path),bytes=size,identity=before,sha256=h.hexdigest(),direct_io=True)
