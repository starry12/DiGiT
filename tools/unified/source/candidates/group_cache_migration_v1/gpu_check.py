"""Tiny CUDA priority/EID/address checks; no dataset load or SSD access."""
import ctypes as C
import os
import tempfile
import numpy as np
from .common import *
def require_accepted():
    r=read(OUT/'gpu_checks.json');require(r['passed'] and r['source_sha256']==verify(),'Current sampler GPU checks required')
    return r

def sparse_checks():
    import subprocess
    from .tests import fixture,native_cpu
    from candidates.ukl_sparse_overlay_v3.native import View,Out,outputs,cuda_call
    lib=C.CDLL(str(OUT/'build/libcompact_cuda.so'));fn=lib.sample_cuda
    fn.argtypes=[View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,Out];fn.restype=C.c_int
    cases=0
    with tempfile.TemporaryDirectory() as td:
        folder=Path(td);info,original,locked,members,a=fixture(folder,False,37);out=folder/'out'
        subprocess.run([str(OUT/'build/reuse'),'build',str(folder/'source.spec'),str(out),'1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
        values=dict(ptr=a['ptr'],idx=a['idx'],gptr=np.fromfile(out/'ptr.i64',np.int64),gidx=np.fromfile(out/'units.i32',np.int32),covered=np.fromfile(out/'covered.i64',np.int64),members=a['members'],primary=np.arange(info['nodes'],dtype=np.int64)+100,bases=np.arange(info['groups'],dtype=np.int64)*2)
        seeds=np.arange(info['nodes'],dtype=np.int32)
        for fan in (1,2,3,5,10,32):
            for grouped in (False,True):
                for seed in (0,23,2**64-1):
                    expected=native_cpu(values,seeds,fan,grouped,seed);actual=outputs(len(seeds),fan)
                    require(cuda_call(fn,values,seeds,fan,grouped,seed,actual)==0,'Sparse CUDA error')
                    for name in expected:np.testing.assert_array_equal(expected[name],actual[name])
                    cases+=1
    return cases

def ig_checks():
    import torch
    from .ig.sampler import load
    mod=load();n=32;ng=16;rng=np.random.default_rng(39)
    degrees=[0,1,2,3,31,32,33,127,2048,4097,8193]+[37]*21
    ptr=np.r_[0,np.cumsum(degrees)].astype(np.int64);ids=rng.integers(0,n+ng,size=int(ptr[-1]),dtype=np.int64)
    ids[ptr[2]:ptr[3]]=n;ids[ptr[3]:ptr[4]]=2
    arrays=[ptr,ids,np.arange(n,dtype=np.int64).reshape(ng,2),np.arange(ng,dtype=np.int64)*4,np.arange(ng,dtype=np.int64),np.arange(n,dtype=np.int64)]
    csc=[np.arange(n+1,dtype=np.int64)*n,np.tile(np.arange(n,dtype=np.int64),n),np.arange(n*n,dtype=np.int64)+2**32]
    seeds=torch.arange(n,dtype=torch.int64,device='cuda');cases=0
    for optimized,cooperative in ((False,False),(True,False),(True,True)):
        mod.set_optimized(optimized);mod.set_cooperative(cooperative);mod.set_bucket(False,-1,2**63-1)
        for compact in (False,True):
            data=[torch.from_numpy(a.astype(np.int64 if i==0 or not compact else np.int32)).cuda() for i,a in enumerate(arrays)]
            original=[torch.from_numpy(a).cuda() for a in csc];fn=mod.sample_group_aware_shared_i64_csc if compact else mod.sample_group_aware
            for fan in (2,3,5,10,31,128):
                for seed in (0,23,2**64-1):
                    src=torch.empty(n*fan,dtype=torch.int64,device='cuda');rows=torch.empty_like(src);flags=torch.empty(n*fan,dtype=torch.uint8,device='cuda');eids=torch.empty_like(src);groups=torch.empty_like(seeds);counts=torch.empty_like(seeds)
                    fn(*[t.data_ptr() for t in data+original],seeds.data_ptr(),n,n,ng,2,fan,seed,*[t.data_ptr() for t in (src,rows,flags,eids,groups,counts)],torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize();ss,rr,ff,ee=[t.cpu().numpy().reshape(n,fan) for t in (src,rows,flags,eids)];gg,cc=groups.cpu().numpy(),counts.cpu().numpy()
                    for owner in range(n):
                        row=ids[ptr[owner]:ptr[owner+1]];g=min(int((row>=n).sum()),fan//2);count=2*g+min(int((row<n).sum()),fan-2*g)
                        require(gg[owner]==g and cc[owner]==count,'IG priority/budget failed')
                        np.testing.assert_array_equal(ff[owner,:2*g],1);np.testing.assert_array_equal(ff[owner,2*g:],0)
                        for j in range(count):
                            eid=int(ee[owner,j])-2**32;node=int(ss[owner,j]);storage=int(rr[owner,j])
                            require(eid//n==owner and eid%n==node,'IG high-bit EID failed')
                            if j<2*g:require(storage%4<2 and node==storage//4*2+storage%4,'IG group address failed')
                            else:require(storage==node,'IG raw address failed')
                    cases+=1
    return cases

def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use free GPU2 for the tiny fixture')
    from candidates.uks_group_reuse_v2.common import exclusive
    from candidates.uks_group_first_cpu2_v1.resources import target_gpu_busy
    with exclusive():
        require(not target_gpu_busy(),'GPU2 is occupied')
        source=verify();sparse=sparse_checks();ig=ig_checks()
        from .ig.adjacency_tests import gpu_fixture
        raw_adjacency=gpu_fixture()
        require(source==verify(),'Source changed during GPU tests')
        write(OUT/'gpu_checks.json',dict(passed=True,source_sha256=source,sparse_cases=sparse,ig_cases=ig,raw_adjacency_cases=raw_adjacency,raw_ssd_access=False,full_graph_training_accepted=False))
if __name__=='__main__':main()
