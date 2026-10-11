"""Tiny CUDA oracle/parity checks, no full graph, NVMe controller or SSD reads."""
import ctypes as C,hashlib,json,os
from pathlib import Path
import numpy as np
from .build import HERE,OUT,ROOT,verify,sha
RECEIPT=OUT.parent/'gpu_checks.json'
def fingerprint():
    paths=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h','.cpp')]
    paths+=[ROOT/'candidates/uks_pa_fast_v1/model.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def require_accepted():
    r=json.loads(RECEIPT.read_text())
    if not r['passed'] or r['sources']!=fingerprint():raise RuntimeError('Fresh fast-path CUDA checks required')
    for k in ('ukl','cl'):
        for role in ('io','sampler'):
            if r['binaries'][k+'_'+role]!=sha(verify(k,role)):raise RuntimeError('CUDA check binary changed')
    return r
def model_check(dim):
    import torch,dgl,copy
    from ae.igb.models import SAGE
    from candidates.uks_pa_fast_v1.model import optimize
    torch.manual_seed(81);a=SAGE(dim,128,19,3,.2).cuda();b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001,weight_decay=0);ob=torch.optim.Adam(b.parameters(),lr=.001,weight_decay=0)
    optimize(b,ob,capture=True,batch_size=4)
    for step,batch in enumerate((4,3,4,3)):
        blocks=[]
        for ns,nd in ((24,16),(16,8),(8,batch)):
            dst=torch.arange(nd,device='cuda').repeat_interleave(2);src=torch.arange(2*nd,device='cuda')%ns
            blocks.append(dgl.create_block((src,dst),num_src_nodes=ns,num_dst_nodes=nd))
        x=torch.randn(24,dim,device='cuda');target=torch.arange(batch,device='cuda')
        for net,opt in ((a,oa),(b,ob)):
            torch.manual_seed(100+step);loss=torch.nn.functional.cross_entropy(net(blocks,x),target)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,rtol=3e-5,atol=3e-6)
            for key in oa.state[p]:torch.testing.assert_close(oa.state[p][key],ob.state[q][key],rtol=3e-5,atol=3e-6)
    if b._capture_calls!=2:raise RuntimeError('Partial batch incorrectly captured')
    return dict(passed=True,feature_dim=dim,updates=4,capture_calls=2,weight_decay=0)
def run():
    from ae.pa_sage.common import setup_imports
    setup_imports()
    import torch
    from candidates.ukl_sparse_overlay_v3.native import View,Out,outputs
    torch.set_num_threads(1);cases=0
    for kind in ('ukl','cl'):
        n=5;origin=2**33
        groups=np.array([0,1,0,1,0],np.int32)
        offsets=np.array([[0,1] if g==0 else [2,3] for g in groups])
        covers=offsets.astype(np.uint32) if kind=='ukl' else offsets+origin+np.arange(n)[:,None]*n
        arrays=dict(ptr=np.arange(n+1,dtype=np.int64)*n+origin,gptr=np.arange(n+1,dtype=np.int64),bases=np.array([0,5],np.int64),
            primary=np.arange(n,dtype=np.int64),covered=covers.reshape(-1),idx=np.tile(np.arange(n,dtype=np.int32),n),gidx=groups,members=np.array([[0,1],[2,3]],np.int32))
        order=('ptr','gptr','bases','primary','covered','idx','gidx','members')
        dev={k:torch.from_numpy(v.copy().view(np.int32) if v.dtype==np.uint32 else v.copy()).cuda() for k,v in arrays.items()}
        from candidates.ukl_compact_cover_v1.build import verified_binary
        cpu_path=verified_binary('libcompact_cpu.so') if kind=='ukl' else ROOT/'results/group_cache_migration_v1/build/libcompact_cpu.so'
        if not cpu_path.exists():
            from candidates.group_cache_migration_v1.build import LARGE_OUT
            cpu_path=LARGE_OUT/'libcompact_cpu.so'
        cpu=C.CDLL(str(cpu_path));fn=cpu.sample_cpu;fn.argtypes=[View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,Out];fn.restype=C.c_int
        lib=C.CDLL(str(verify(kind,'sampler')));fast=lib.sample_stream
        fast.argtypes=[View,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,C.c_int64,Out,C.c_void_p,C.c_uint64];fast.restype=C.c_int
        stream=torch.cuda.Stream()
        for grouped in (False,True):
            for f in (1,2,3,5,10,32):
                for seed in (0,23,2**64-1):
                    seeds=np.arange(n,dtype=np.int32);ref=outputs(n,f)
                    ptrs=[arrays[k].ctypes.data if k!='primary' or grouped else None for k in order]
                    view=View(n,n*n,origin,n,0,2,*ptrs)
                    if fn(view,seeds.ctypes.data,n,f,int(grouped),seed,Out(*[v.ctypes.data for v in ref.values()])):raise RuntimeError('CPU fixture failed')
                    with torch.cuda.stream(stream):
                        s=torch.from_numpy(seeds).cuda();out={k:torch.full(v.shape,-1,dtype=torch.int64 if v.dtype==np.int64 else torch.int32,device='cuda') for k,v in ref.items()};g=torch.zeros(n,dtype=torch.int64,device='cuda')
                        view=View(n,n*n,origin,n,0,2,*[dev[k].data_ptr() if k!='primary' or grouped else None for k in order])
                        rc=fast(view,s.data_ptr(),n,f,int(grouped),seed,7 if grouped else n,Out(*[v.data_ptr() for v in out.values()]),g.data_ptr(),stream.cuda_stream)
                    stream.synchronize()
                    if rc:raise RuntimeError('Fast fixture launch failed')
                    for k in ref:np.testing.assert_array_equal(ref[k],out[k].cpu().numpy())
                    cases+=1
        address=lib.annotate_stream
        address.argtypes=[View,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int64,C.c_int,C.c_int64,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint64];address.restype=C.c_int
        with torch.cuda.stream(stream):
            ids=torch.tensor([0,1,2,3,4],dtype=torch.int64,device='cuda')
            first=torch.tensor([0,1,2,3,4],dtype=torch.int64,device='cuda')
            rows=torch.tensor([0,1,5,6,4],dtype=torch.int64,device='cuda')
            result=torch.empty_like(ids);heads=torch.empty_like(ids);errors=torch.zeros(1,dtype=torch.int32,device='cuda')
            rc=address(view,ids.data_ptr(),first.data_ptr(),rows.data_ptr(),5,5,7,1,result.data_ptr(),heads.data_ptr(),errors.data_ptr(),stream.cuda_stream)
        stream.synchronize()
        if rc or int(errors.item()) or result.tolist()!=[0,1,5,6,4] or heads.tolist()!=[0,0,5,5,-1]:raise RuntimeError('Odd replica head/alias annotation mismatch')
        with torch.cuda.stream(stream):
            rows[2]=6;errors.zero_()
            rc=address(view,ids.data_ptr(),first.data_ptr(),rows.data_ptr(),5,5,7,1,result.data_ptr(),heads.data_ptr(),errors.data_ptr(),stream.cuda_stream)
        stream.synchronize()
        if rc or not int(errors.item()):raise RuntimeError('Wrong replica alias accepted')
        # Invalid owner and malformed covered order must be rejected on-device.
        for bad in ('owner','cover'):
            with torch.cuda.stream(stream):
                s=torch.arange(n,dtype=torch.int32,device='cuda');g=torch.zeros(n,dtype=torch.int64,device='cuda')
                if bad=='owner':s[0]=-1
                else:dev['covered'][1]=dev['covered'][0]
                rc=fast(view,s.data_ptr(),n,32,1,0,7,Out(*[v.data_ptr() for v in out.values()]),g.data_ptr(),stream.cuda_stream)
            stream.synchronize()
            if rc or not int(out['errors'][0]):raise RuntimeError('Corrupt owner/cover accepted')
    models=[model_check(128),model_check(1024)]
    return dict(passed=True,sources=fingerprint(),binaries={k+'_'+r:sha(verify(k,r)) for k in ('ukl','cl') for r in ('io','sampler')},
        sampler_cases=cases,model_checks=models,raw_ssd_access=False,large_graph_loaded=False,full_graph_training_accepted=False)
def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('Select GPU2 explicitly')
    from candidates.uks_group_reuse_v2.common import exclusive
    from candidates.uks_group_first_cpu2_v1.resources import target_gpu_busy
    with exclusive():
        if target_gpu_busy():raise RuntimeError('GPU2 occupied; defer CUDA checks')
        result=run();RECEIPT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
