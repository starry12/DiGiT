"""Explicit tiny native priority/parity check; no UKS load or SSD I/O."""
import os
from .common import *

def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Set CUDA_VISIBLE_DEVICES=2')
    from .resources import target_gpu_busy
    with exclusive():
        require(not target_gpu_busy(),'Target GPU2 is occupied')
        from candidates.uks_mixed_1k2k_v1.runtime import setup_sampling_imports
        setup_sampling_imports()
        import numpy as np,torch
        mod=extension();n=32;ng=16;rng=np.random.default_rng(39)
        degrees=[0,1,2,3,31,32,33,127,2048]+[37]*23
        ptr=np.r_[0,np.cumsum(degrees)].astype(np.int64)
        ids=rng.integers(0,n+ng,size=int(ptr[-1]),dtype=np.int64)
        # Explicit no-ordinary, no-group and repeated-group cases.
        ids[ptr[2]:ptr[3]]=n
        ids[ptr[3]:ptr[4]]=2
        arrays=[ptr,ids,np.arange(n,dtype=np.int64).reshape(ng,2),
                np.arange(ng,dtype=np.int64)*4,np.arange(ng,dtype=np.int64),np.arange(n,dtype=np.int64)]
        csc=[np.arange(n+1,dtype=np.int64)*n,np.tile(np.arange(n,dtype=np.int64),n),np.arange(n*n,dtype=np.int64)+2**32]
        seeds=torch.arange(n,dtype=torch.int64,device='cuda');cases=0;reference={}
        for dtype in (np.int64,np.int32):
            data=[torch.from_numpy(a.astype(np.int64 if i==0 else dtype)).cuda() for i,a in enumerate(arrays)]
            original=[torch.from_numpy(a).cuda() for a in csc]
            fn=mod.sample_group_aware if dtype==np.int64 else mod.sample_group_aware_i32_uva64
            for fan in (1,2,3,5,10,31,128):
                for seed in (0,23,2**64-1):
                    src=torch.empty(n*fan,dtype=torch.int64,device='cuda');rows=torch.empty_like(src)
                    flags=torch.empty(n*fan,dtype=torch.uint8,device='cuda');eids=torch.empty_like(src)
                    groups=torch.empty_like(seeds);nodes=torch.empty_like(seeds)
                    fn(*[t.data_ptr() for t in data+original],seeds.data_ptr(),n,n,ng,2,fan,seed,
                       *[t.data_ptr() for t in (src,rows,flags,eids,groups,nodes)],torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize()
                    ss,rr,ff,ee=[t.cpu().numpy().reshape(n,fan) for t in (src,rows,flags,eids)]
                    gc,nc=groups.cpu().numpy(),nodes.cpu().numpy()
                    values=[ss,rr,ff,ee,gc,nc]
                    if dtype==np.int64:reference[(fan,seed)]=[v.copy() for v in values]
                    else:
                        for a,b in zip(reference[(fan,seed)],values):np.testing.assert_array_equal(a,b)
                    for owner in range(n):
                        row=ids[ptr[owner]:ptr[owner+1]];want_g=min(int((row>=n).sum()),fan//2)
                        want_n=2*want_g+min(int((row<n).sum()),fan-2*want_g)
                        require(gc[owner]==want_g and nc[owner]==want_n,'Priority/budget mismatch')
                        np.testing.assert_array_equal(ff[owner,:2*want_g],1)
                        np.testing.assert_array_equal(ff[owner,2*want_g:],0)
                        np.testing.assert_array_equal(ss[owner,want_n:],-1)
                        for slot in range(want_n):
                            eid=int(ee[owner,slot])-2**32;node=int(ss[owner,slot]);storage=int(rr[owner,slot])
                            require(eid//n==owner and eid%n==node,'Incorrect/high-bit-truncated EID')
                            if slot<2*want_g:
                                require(storage%4<2 and node==(storage//4)*2+storage%4,'Group feature row mismatch')
                                if slot%2==0:require(rr[owner,slot+1]==storage+1,'Split group emission')
                            else:require(storage==node,'Ordinary feature row mismatch')
                    cases+=1
        write(OUT/'gpu_checks.json',dict(passed=True,source_sha256=verify(),cases=cases,
              selection_policy='groups_first_without_replacement',native_training_accepted=False,
              raw_ssd_access=False,eid_above_32bit=True))

if __name__=='__main__':main()
