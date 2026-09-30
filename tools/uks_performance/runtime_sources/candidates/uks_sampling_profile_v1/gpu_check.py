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
    mod=extension();n=32
    raw=np.array([(i,(i+j)%n) for i in range(n) for j in (0,1,2,3,7,11)],dtype=np.int64).T
    ptr,idx,eid=normalized_fixture(raw,n);g=graph_from_csc(ptr,idx,eid,n,fixture=True);g.pin_memory_()
    with tempfile.TemporaryDirectory() as tmp:
        bundle=reorganize_to_bundle(ptr,idx,np.zeros((n,256),np.float32),Path(tmp)/'layout',dataset_name='UKS',dataset_size='fixture',group_size=2,replication_ratio=.2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096,hot_nodes=np.array([0,1],np.int64),seed=0)
        s=DiGiTNeighborSampler([10,5,5],bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=0);meta=s._ensure_cuda_metadata(g,torch.device('cuda:0'))
        high_eids=(torch.from_numpy(eid.copy())+2**32).pin_memory();meta['original_eids']=high_eids
        seeds=torch.tensor([0,7,15,31],device='cuda',dtype=torch.int64);fan=10;reference=None;mod.profile_prepare(1,torch.cuda.current_stream().cuda_stream)
        for m in (_cuda_extension,mod):
            out=[torch.empty(len(seeds)*fan,dtype=torch.int64,device='cuda') for _ in range(3)]
            sources,rows,eids=out;flags=torch.empty_like(sources,dtype=torch.uint8);groups=torch.empty_like(seeds);nodes=torch.empty_like(seeds)
            if m is mod:mod.profile_arm(1)
            m.sample_group_aware_i32_uva64(*[meta[k].data_ptr() for k in ('reorganized_indptr','reorganized_indices','group_members','group_storage_base','supernode_to_group','node_to_primary','original_indptr','original_indices','original_eids')],seeds.data_ptr(),len(seeds),n,bundle.num_groups,2,fan,23,sources.data_ptr(),rows.data_ptr(),flags.data_ptr(),eids.data_ptr(),groups.data_ptr(),nodes.data_ptr(),torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize();values=[x.cpu().numpy().copy() for x in (sources,rows,flags,eids,groups,nodes)]
            if reference is None:reference=values
            else:require(all(np.array_equal(a,b) for a,b in zip(reference,values)),'Instrumented kernel output differs')
            require(np.all(values[3][values[0]>=0]>=2**32),'EIDs truncated')
        events=[dict(x) for x in mod.profile_collect()];mod.profile_release();require(len(events)==1,'Missing events')
    write(OUT/'gpu_check/report.json',dict(passed=True,outputs_bit_exact=True,eid_above_32bit=True,events=events,source_sha256=verify(),raw_ssd_writes=False))
if __name__=='__main__':main()
