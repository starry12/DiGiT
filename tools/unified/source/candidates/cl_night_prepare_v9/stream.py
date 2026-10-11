"""Small aligned direct reads into owned buffers; no source file mmap."""
import hashlib,mmap,os,time
from pathlib import Path
import numpy as np
from candidates.ukl_runtime_prepare_v11r1 import data as D

def read_chunks(info,dtype,check=lambda:None,chunk=8*2**20,rate=512*2**20):
    p=Path(info['path']);length=info.get('length',info.get('bytes'))
    if p.is_symlink() or D.ident(p)!=info['identity']:raise RuntimeError('Read source changed')
    if chunk<=0 or chunk%4096 or length%np.dtype(dtype).itemsize:raise ValueError('Direct chunk alignment')
    fd=os.open(str(p),os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW|os.O_CLOEXEC)
    buf=mmap.mmap(-1,chunk,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
    digest=hashlib.sha256();start=time.monotonic()
    try:
        for lo in range(0,length,chunk):
            check();size=min(chunk,length-lo);aligned=(size+4095)//4096*4096
            view=memoryview(buf)[:aligned]
            try:got=os.preadv(fd,[view],lo)
            finally:view.release()
            if got!=size:raise IOError('Short direct input')
            raw=buf[:size];digest.update(raw)
            value=np.frombuffer(raw,dtype=dtype).copy()
            while (lo+size)/rate>time.monotonic()-start:
                check();time.sleep(min(.2,max(0,(lo+size)/rate-(time.monotonic()-start))))
            yield lo//np.dtype(dtype).itemsize,value
        if digest.hexdigest()!=info['sha256'] or D.ident(p)!=info['identity']:
            raise RuntimeError('Read source hash/identity changed')
    finally:buf.close();os.close(fd)

def inverse_mapping(specs,n,rows,check=lambda:None,progress=lambda *a:None):
    mapping=np.full(rows,-1,dtype=np.int32)
    for lo,physical in read_chunks(specs['primary'],np.int64,check):
        if np.any(physical<0) or np.any(physical>=n):raise ValueError('Primary range')
        mapping[physical]=np.arange(lo,lo+len(physical),dtype=np.int32);progress('primary',lo+len(physical),n)
    bases=read_chunks(specs['bases'],np.int64,check)
    members=read_chunks(specs['members'],np.int32,check)
    # Both records have width 8 bytes: paired chunks have identical row counts.
    try:
        while True:
            a=next(bases,None);b=next(members,None)
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError('Group array length mismatch')
                break
            lo,base=a;offset,pair=b;pair=pair.reshape(-1,2)
            if offset!=lo*2 or len(base)!=len(pair):raise ValueError('Group chunk mismatch')
            if np.any(base<0) or np.any(base+1>=rows) or np.any(pair<0) or np.any(pair>=n):raise ValueError('Group range')
            primary=base<n
            if not np.array_equal(mapping[base[primary]],pair[primary,0]) or not np.array_equal(mapping[base[primary]+1],pair[primary,1]):raise ValueError('Primary group mismatch')
            mapping[base[~primary]]=pair[~primary,0];mapping[base[~primary]+1]=pair[~primary,1]
            progress('groups',lo+len(base),specs['bases'].get('length',specs['bases'].get('bytes'))//8)
    finally:bases.close();members.close()
    for lo,physical in read_chunks(specs['primary'],np.int64,check):
        if not np.array_equal(mapping[physical],np.arange(lo,lo+len(physical),dtype=np.int32)):raise ValueError('Primary collision')
    for lo in range(0,rows,1048576):
        check();part=mapping[lo:lo+1048576]
        if np.any(part<0) or np.any(part>=n):raise ValueError('Unmapped physical row')
    return mapping

def mapped_feature_chunks(info,n,check=lambda:None):
    for _,ids in read_chunks(info,np.int32,check):
        for lo in range(0,len(ids),16384):
            check();logical=ids[lo:lo+16384]
            if np.any(logical<0) or np.any(logical>=n):raise ValueError('Logical row range')
            yield D.features(logical)

def natural_feature_chunks(n,check=lambda:None):
    for lo in range(0,n,16384):
        check();yield D.features(np.arange(lo,min(n,lo+16384),dtype=np.int64))
