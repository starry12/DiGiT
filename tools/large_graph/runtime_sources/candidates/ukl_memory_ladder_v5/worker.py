"""One small synthetic extent: file -> anonymous host -> GPU -> bounded readback."""
import ctypes as C
import hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
from .memory import Arena,Registration,RETAINED,CHUNK
from .limits import effective_limits
from .protocol import validate_tier,budget

def source(path,size):
    h=hashlib.sha256();started=time.monotonic()
    with path.open('xb') as f:
        for offset in range(0,size,CHUNK):
            count=min(CHUNK,size-offset)//8
            data=(np.arange(count,dtype=np.int64)+(offset//8)).tobytes()
            f.write(data);h.update(data)
            done=offset+len(data)
            if done % (16*1024**2)==0:f.flush();os.fsync(f.fileno())
            wait=done/(64*1024**2)-(time.monotonic()-started)
            if wait>0:time.sleep(wait)
        f.flush();os.fsync(f.fileno())
    return h.hexdigest()

def gpu_readback(arena,registration,progress):
    api=registration.api;rt=api.rt
    for name,args in {'cudaMalloc':[C.POINTER(C.c_void_p),C.c_size_t],'cudaMemcpy':[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int],'cudaFree':[C.c_void_p]}.items():
        getattr(rt,name).argtypes=args;getattr(rt,name).restype=C.c_int
    device=C.c_void_p();api.call('cudaMalloc',C.byref(device),CHUNK)
    host=(C.c_char*CHUNK)();h=hashlib.sha256();total=arena.size
    try:
        for offset in range(0,total,CHUNK):
            n=min(CHUNK,total-offset)
            # Registered mapped-host device pointer -> GPU buffer -> CPU buffer.
            api.call('cudaMemcpy',device,registration.pointers['data']+offset,n,3)
            api.call('cudaMemcpy',C.addressof(host),device,n,2)
            api.synchronize();data=bytes(host[:n])
            if data!=arena.mm[offset:offset+n]:raise RuntimeError('GPU readback mismatch at '+str(offset))
            h.update(data);progress(offset+n)
    finally:
        api.synchronize();api.call('cudaFree',device)
    return h.hexdigest()

def process_memory():
    keys={'VmRSS','VmLck','VmPin','RssAnon','RssFile'}
    return {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.split(':')[0] in keys}

def unmapped(address,size):
    for line in Path('/proc/self/maps').read_text().splitlines():
        lo,hi=(int(x,16) for x in line.split()[0].split('-'))
        if max(lo,address)<min(hi,address+size):return False
    return True

def main():
    if os.environ.get('UKL_V6_BOUNDED_WORKER')!='1':raise RuntimeError('Use bounded controller')
    tier=int(os.environ['UKL_MEMORY_TIER_MIB']);size=validate_tier(tier);out=Path(os.environ['UKL_V6_OUTPUT'])
    limits=effective_limits(tier);(out/'effective_limits.json').write_text(json.dumps(limits,indent=2)+'\n')
    events=[]
    def event(stage,**kw):
        events.append(dict(stage=stage,time=time.time(),process_memory=process_memory(),**kw));(out/'lifecycle.json').write_text(json.dumps(events,indent=2)+'\n')
    event('source');p=Path(os.environ['UKL_SOURCE_DIR'])/'synthetic.i64';expected=source(p,size)
    arena=None;registration=None
    try:
        event('load');arena=Arena({'data':(p,np.int64,(size//8,),expected)}).load()
        event('loaded',arena_bytes=arena.size);address=arena.address
        event('register');registration=Registration(arena)
        event('registered')
        event('gpu_readback');actual=gpu_readback(arena,registration,lambda n:event('gpu_progress',bytes=n) if n%(256*1024**2)==0 or n==size else None)
        if actual!=expected:raise RuntimeError('Readback digest mismatch')
        event('unregister');registration.close();registration=None
        event('unregistered');event('release');arena.close();arena=None
        if not unmapped(address,size):raise RuntimeError('Anonymous VMA still present after close')
        event('released',vma_absent=True)
        if RETAINED:raise RuntimeError('Unreleased registration')
        event('complete')
        (out/'worker.json').write_text(json.dumps(dict(passed=True,tier_mib=tier,loaded_bytes=size,gpu_readback_bytes=size,source_sha256=expected,gpu_sha256=actual,limits_verified_before_cuda=True,effective_limits=limits,normal_unregister=True,anonymous_memory_released=True,vma_absent_after_release=True,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,full_graph=False,raw_ssd_access=False),indent=2)+'\n')
    finally:
        if registration:registration.close()
        if arena:arena.close()
if __name__=='__main__':main()
