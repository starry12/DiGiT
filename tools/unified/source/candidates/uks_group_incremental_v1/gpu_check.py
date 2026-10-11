"""Compare original and event-instrumented native outputs, including >32bit EIDs."""
from .common import *
def main():
    check_ready()
    from candidates.uks_native_v1.runtime import setup_sampling_imports
    setup_sampling_imports()
    import numpy as np,torch,dgl,tempfile
    from digit.sampler import DiGiTNeighborSampler,_cuda_extension
    from digit.reorganization import reorganize_to_bundle
    from candidates.uks_native_v1.graph import normalized_fixture,graph_from_csc
    mod=raw_extension();n=32
    raw=np.array([(i,(i+j)%n) for i in range(n) for j in (0,1,2,3,7,11)],dtype=np.int64).T
    ptr,idx,eid=normalized_fixture(raw,n);g=graph_from_csc(ptr,idx,eid,n,fixture=True);g.pin_memory_()
    with tempfile.TemporaryDirectory() as tmp:
        bundle=reorganize_to_bundle(ptr,idx,np.zeros((n,256),np.float32),Path(tmp)/'layout',dataset_name='UKS',dataset_size='fixture',group_size=2,replication_ratio=.2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096,hot_nodes=np.array([0,1],np.int64),seed=0)
        s=DiGiTNeighborSampler([10,5,5],bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=0);meta=s._ensure_cuda_metadata(g,torch.device('cuda:0'))
        high_eids=(torch.from_numpy(eid.copy())+2**32).pin_memory();meta['original_eids']=high_eids
        seeds=torch.tensor([0,7,15,31],device='cuda',dtype=torch.int64);fan=10;reference=None;mod.profile_prepare(2,torch.cuda.current_stream().cuda_stream)
        for case,m in enumerate((_cuda_extension,mod,mod)):
            out=[torch.empty(len(seeds)*fan,dtype=torch.int64,device='cuda') for _ in range(3)]
            sources,rows,eids=out;flags=torch.empty_like(sources,dtype=torch.uint8);groups=torch.empty_like(seeds);nodes=torch.empty_like(seeds)
            if m is mod:mod.profile_arm(case)
            fn=mod.sample_group_aware_i32_uva64_incremental if case==2 else m.sample_group_aware_i32_uva64
            fn(*[meta[k].data_ptr() for k in ('reorganized_indptr','reorganized_indices','group_members','group_storage_base','supernode_to_group','node_to_primary','original_indptr','original_indices','original_eids')],seeds.data_ptr(),len(seeds),n,bundle.num_groups,2,fan,23,sources.data_ptr(),rows.data_ptr(),flags.data_ptr(),eids.data_ptr(),groups.data_ptr(),nodes.data_ptr(),torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize();values=[x.cpu().numpy().copy() for x in (sources,rows,flags,eids,groups,nodes)]
            if reference is None:reference=values
            else:require(all(np.array_equal(a,b) for a,b in zip(reference,values)),'Instrumented kernel output differs')
            require(np.all(values[3][values[0]>=0]>=2**32),'EIDs truncated')
        events=[dict(x) for x in mod.profile_collect()];mod.profile_release();require(len(events)==2,'Missing events')
    cases=stress(mod,torch,np)
    write(OUT/'gpu_check/report.json',dict(passed=True,outputs_bit_exact=True,stress_cases=cases,eid_above_32bit=True,events=events,source_sha256=verify(),raw_ssd_writes=False))
def stress(mod,torch,np):
    # Synthetic high-indegree rows with repeated node/group occurrences.
    n=32;ng=16;rng=np.random.RandomState(23)
    degrees=[0,1,2,3,31,32,33,63,127,1024,10000,100000]+[73]*20
    ptr=np.concatenate(([0],np.cumsum(degrees))).astype(np.int64)
    idx=rng.randint(0,n+ng,size=int(ptr[-1])).astype(np.int32)
    tensors=[torch.from_numpy(a).cuda() for a in [ptr,idx,np.arange(n,dtype=np.int32).reshape(ng,2),np.arange(ng,dtype=np.int32)*4,np.arange(ng,dtype=np.int32),np.arange(n,dtype=np.int32)]]
    # Real CSC EID lookup, values deliberately above 2**32.
    csc=[torch.from_numpy(a).pin_memory() for a in [np.arange(n+1,dtype=np.int64)*n,np.tile(np.arange(n,dtype=np.int64),n),np.arange(n*n,dtype=np.int64)+2**32]]
    seeds=torch.arange(n,device='cuda',dtype=torch.int64);cases=0
    for fan in (2,3,5,10,31,128):
        for seed in (0,23,2**64-1):
            outputs=[]
            for fn in (mod.sample_group_aware_i32_uva64,mod.sample_group_aware_i32_uva64_incremental):
                src=torch.empty(n*fan,dtype=torch.int64,device='cuda');rows=torch.empty_like(src);flags=torch.empty_like(src,dtype=torch.uint8);eid=torch.empty_like(src);groups=torch.empty_like(seeds);nodes=torch.empty_like(seeds)
                fn(*[t.data_ptr() for t in tensors+csc],seeds.data_ptr(),n,n,ng,2,fan,seed,*[t.data_ptr() for t in (src,rows,flags,eid,groups,nodes)],torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize();outputs.append([t.cpu().numpy().copy() for t in (src,rows,flags,eid,groups,nodes)])
            require(all(np.array_equal(a,b) for a,b in zip(*outputs)),'Stress parity failed fan=%d seed=%d'%(fan,seed));cases+=1
    return cases
if __name__=='__main__':main()
