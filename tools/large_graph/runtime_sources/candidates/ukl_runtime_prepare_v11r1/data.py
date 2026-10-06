"""Deterministic protocol and bounded direct-file publication (no CUDA)."""
import ctypes as C
import hashlib
import mmap
import os
from pathlib import Path
import time
import numpy as np

PAGE=4096
CHUNK=8*1024**2

def ident(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def mix(x):
    x=np.asarray(x,dtype=np.uint64).copy()
    with np.errstate(over='ignore'):
        x+=np.uint64(0x9e3779b97f4a7c15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        x^=x>>np.uint64(31)
    return x

def features(ids):
    ids=np.asarray(ids,dtype=np.uint64)
    return ((mix(ids[:,None]*np.uint64(128)+np.arange(128,dtype=np.uint64)[None,:]
                 +np.uint64(20260922))>>np.uint64(40)).astype(np.float32)
            *np.float32(2./16777216)-np.float32(1))

def labels(ids):
    return (mix(np.asarray(ids,dtype=np.uint64)+np.uint64(20260923))%np.uint64(19)).astype(np.int64)

def split(n,batch=1024,window=320,freq_batches=100):
    count=n//10
    if count<max(window,freq_batches)*batch:raise ValueError('training split too small')
    train=np.sort(np.random.default_rng(0).choice(n,count,replace=False)).astype(np.int64)
    order=np.random.default_rng(0).permutation(count)
    gids=train[order[:window*batch]].copy();del order
    order=np.random.default_rng(23).permutation(count)
    freq=train[order[:freq_batches*batch]].copy()
    return train,gids,freq

def bfs_window(order,batch=1024,count=320):
    blocks=len(order)//batch
    if blocks<count:raise ValueError('BFS window too short')
    choice=np.random.default_rng(np.random.SeedSequence([0,0])).permutation(blocks)[:count]
    return np.concatenate([order[int(i)*batch:(int(i)+1)*batch] for i in choice]).astype(np.int64)

def inverse_map(arrays,n,rows,check=lambda:None,chunk=1048576):
    mapping=np.full(rows,-1,dtype=np.int32)
    for lo in range(0,n,chunk):
        check();hi=min(n,lo+chunk);physical=arrays['primary'][lo:hi]
        if np.any(physical<0) or np.any(physical>=n):raise ValueError('primary range')
        mapping[physical]=np.arange(lo,hi,dtype=np.int32)
    for lo in range(0,len(arrays['bases']),chunk):
        check();base=arrays['bases'][lo:lo+chunk];members=arrays['members'][lo:lo+chunk]
        if np.any(base<0) or np.any(base+1>=rows) or np.any(members<0) or np.any(members>=n):
            raise ValueError('group range')
        primary=base<n
        if not np.array_equal(mapping[base[primary]],members[primary,0]) or not np.array_equal(mapping[base[primary]+1],members[primary,1]):
            raise ValueError('primary group mismatch')
        mapping[base[~primary]]=members[~primary,0];mapping[base[~primary]+1]=members[~primary,1]
    for lo in range(0,n,chunk):
        check();hi=min(n,lo+chunk)
        if not np.array_equal(mapping[arrays['primary'][lo:hi]],np.arange(lo,hi,dtype=np.int32)):
            raise ValueError('primary collision')
    for lo in range(0,rows,chunk):
        check();values=mapping[lo:lo+chunk]
        if np.any(values<0) or np.any(values>=n):raise ValueError('unmapped row')
    return mapping

def direct_file(path,chunks,check=lambda:None,rate=512*1024**2,progress=lambda n:None):
    """Exclusive .partial, aligned O_DIRECT, bounded staging, durable rename.

    No buffered fallback. Failed partial files remain for diagnosis; caller must
    use a new output directory. Hash covers logical bytes, not final page padding.
    """
    path=Path(path);partial=path.with_name(path.name+'.partial')
    if path.exists() or path.is_symlink():raise FileExistsError(path)
    if rate<=0:raise ValueError('positive write rate required')
    fd=os.open(str(partial),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_DIRECT,0o644)
    buf=mmap.mmap(-1,CHUNK,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
    digest=hashlib.sha256();logical=physical=used=0;started=time.monotonic()
    def emit(length):
        nonlocal physical
        check();view=memoryview(buf)[:length]
        try:
            got=os.pwritev(fd,[view],physical)
            if got!=length:raise IOError('short direct write')
        finally:view.release()
        physical+=length
        # Bound both in-flight data and dirty page cache; direct writes already
        # completed before the buffer is reused. fsync is scoped to this fd.
        while physical/rate>time.monotonic()-started:
            check();time.sleep(max(0,min(0.2,physical/rate-(time.monotonic()-started))))
        progress(logical)
    try:
        for value in chunks:
            check();data=memoryview(value).cast('B')
            try:
                pos=0
                while pos<len(data):
                    check();take=min(CHUNK-used,len(data)-pos)
                    buf[used:used+take]=data[pos:pos+take];digest.update(data[pos:pos+take])
                    pos+=take;used+=take;logical+=take
                    if used==CHUNK:emit(CHUNK);used=0
            finally:data.release()
        if used:
            padded=(used+PAGE-1)//PAGE*PAGE
            buf[used:padded]=bytes(padded-used);emit(padded)
        check();os.ftruncate(fd,logical);os.fsync(fd)
    finally:
        os.close(fd);buf.close()
    # Hard-link publication refuses an existing destination even under races.
    os.link(str(partial),str(path),follow_symlinks=False);partial.unlink()
    dfd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(dfd)
    finally:os.close(dfd)
    return dict(path=str(path),bytes=logical,sha256=digest.hexdigest(),identity=ident(path),direct_io=True)

def array_chunks(array):
    raw=memoryview(array).cast('B')
    try:
        for lo in range(0,len(raw),CHUNK):yield raw[lo:lo+CHUNK]
    finally:raw.release()

class Algorithms:
    CALLBACK=C.CFUNCTYPE(C.c_int,C.c_uint64,C.c_uint64,C.c_int)
    def __init__(self,path):
        self.lib=C.CDLL(str(path))
        self.lib.bfs.argtypes=[C.c_void_p,C.c_void_p,C.c_int64,C.c_int64,C.c_void_p,
                              C.c_int64,C.c_void_p,C.c_uint64,self.CALLBACK,C.c_void_p,C.c_void_p,C.c_size_t]
        self.lib.bfs.restype=C.c_int
        self.lib.topk.argtypes=[C.c_void_p,C.c_int64,C.c_void_p,self.CALLBACK,C.c_void_p,C.c_size_t]
        self.lib.topk.restype=C.c_int
    def invoke(self,name,args,check):
        failure=[]
        def tick(done,total,stage):
            try:check(done,total,stage);return 0
            except BaseException as error:failure.append(error);return 1
        cb=self.CALLBACK(tick);err=C.create_string_buffer(512)
        rc=getattr(self.lib,name)(*args,cb,err,len(err))
        if failure:raise failure[0]
        if rc:raise RuntimeError(err.value.decode())
    def topk(self,counts,check=lambda *args:None):
        if counts.dtype!=np.uint64 or counts.ndim!=1 or not counts.flags.c_contiguous:raise ValueError('counts')
        output=np.empty(len(counts)//10,np.int64)
        self.invoke('topk',[counts.ctypes.data,len(counts),output.ctypes.data],check)
        return output
    def bfs(self,ptr,idx,train,check=lambda *args:None,edge_cap=(64*1024**3)//4):
        for a,dtype in [(ptr,np.int64),(idx,np.int32),(train,np.int64)]:
            if a.dtype!=dtype or a.ndim!=1 or not a.flags.c_contiguous:raise ValueError('bfs array')
        output=np.empty(len(train),np.int64);stats=np.empty(2,np.uint64)
        failure=[]
        def tick(done,total,stage):
            try:check(done,total,stage);return 0
            except BaseException as error:failure.append(error);return 1
        cb=self.CALLBACK(tick);err=C.create_string_buffer(512)
        rc=self.lib.bfs(ptr.ctypes.data,idx.ctypes.data,len(ptr)-1,len(idx),train.ctypes.data,
                       len(train),output.ctypes.data,edge_cap,cb,stats.ctypes.data,err,len(err))
        if failure:raise failure[0]
        if rc:raise RuntimeError(err.value.decode())
        return output,dict(undirected_induced_edges=int(stats[0]),components=int(stats[1]))
