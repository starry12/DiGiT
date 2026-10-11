"""Fresh compact-ABI parity gate; anonymous tiny inputs, no NVMe access."""
import ctypes as C,json,os
import numpy as np
from .build import verified_binary,OUT,ROOT
from .provenance import verify
from candidates.ukl_sparse_overlay_v3.native import View,Out,outputs,cuda_call
def require_accepted():
    r=json.loads((OUT.parent/'gpu_checks.json').read_text())
    if not r['passed'] or r['source_sha256']!=verify():raise RuntimeError('Current compact CUDA parity required')
    return r
def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('Select GPU2 explicitly')
    from candidates.uks_group_reuse_v2.common import exclusive
    from candidates.uks_group_first_cpu2_v1.resources import target_gpu_busy
    with exclusive():
        if target_gpu_busy():raise RuntimeError('GPU2 is occupied')
        source=verify();n=4;origin=2**33
        a=dict(ptr=np.arange(n+1,dtype=np.int64)*4+origin,gptr=np.arange(n+1,dtype=np.int64),
               bases=np.array([0],np.int64),primary=np.arange(n,dtype=np.int64),covered=np.tile(np.array([0,1],np.uint32),n),
               idx=np.tile(np.arange(n,dtype=np.int32),n),gidx=np.zeros(n,np.int32),members=np.array([[0,1]],np.int32))
        order=('ptr','gptr','bases','primary','covered','idx','gidx','members');seeds=np.arange(n,dtype=np.int32);fns=[]
        for gpu in (False,True):
            lib=C.CDLL(str(verified_binary('libcompact_cuda.so' if gpu else 'libcompact_cpu.so')))
            fn=lib.sample_cuda if gpu else lib.sample_cpu;fn.argtypes=[View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,Out];fn.restype=C.c_int
            fns.append((lib,fn))
        cases=0
        for grouped in (False,True):
            for fanout in (1,2,3,5,10,32):
                for seed in (0,23,2**64-1):
                    cpu=outputs(n,fanout);gpu=outputs(n,fanout)
                    v=View(n,n*n,origin,n,0,1,*[a[k].ctypes.data for k in order])
                    rc=fns[0][1](v,seeds.ctypes.data,n,fanout,int(grouped),seed,Out(*[x.ctypes.data for x in cpu.values()]))
                    grc=cuda_call(fns[1][1],a,seeds,fanout,grouped,seed,gpu)
                    if rc or grc or cpu['errors'].any() or gpu['errors'].any():raise RuntimeError('Compact sampler error')
                    for k in cpu:np.testing.assert_array_equal(cpu[k],gpu[k])
                    cases+=1
        if source!=verify():raise RuntimeError('Source changed')
        (OUT.parent/'gpu_checks.json').write_text(json.dumps(dict(passed=True,source_sha256=source,cases=cases,raw_ssd_access=False,full_graph_training_accepted=False),indent=2))
if __name__=='__main__':main()
