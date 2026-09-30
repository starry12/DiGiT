"""Exact multigraph overlay, mixed-width UVA CUDA parity and overflow regression."""
import argparse,tempfile
import numpy as np
from candidates.pa_sage_bidir_native_v2.common import *
from candidates.pa_sage_bidir_native_v2.overlay import rewrite,exact_validate,apply

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();setup()
    import torch,dgl,runner as r
    from candidates.pa_sage_direction_v2.graph import build,verify_sample
    from digit.reorganization import reorganize_to_bundle
    from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP
    from digit import DiGiTSamplerCUDA as cuda
    require(Path(cuda.__file__).resolve()==(HERE/'runtime/digit/DiGiTSamplerCUDA.so').resolve(),'Wrong sampler extension imported')
    r.startup();n=16;edges=np.array([(u,v) for v in range(n) for u in range(n) if u!=v and (u+2*v)%3==0]+[(0,1),(0,1)],dtype=np.int64)
    direct=build(edges,n,'directed');g=build(edges,n,'bidirectional');op,oi,_=[t.numpy() for t in direct.adj_tensors('csc')];bp,bi,be=[t.numpy() for t in g.adj_tensors('csc')]
    features=np.random.default_rng(3).normal(size=(n,128)).astype('float32')
    with tempfile.TemporaryDirectory(prefix='digit-bidir-fixture-') as tmp:
        base=reorganize_to_bundle(op,oi,features,tmp,dataset_name='fixture',dataset_size='tiny',group_size=2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096)
        require(base.num_groups>0,'Fixture did not exercise groups')
        rp=np.empty_like(base.arrays['reordered_indptr']);ri=np.empty(len(base.arrays['reordered_indices'])+len(edges),dtype=np.int64)
        rewrite(base.arrays['reordered_indptr'],base.arrays['reordered_indices'],op,bp,bi,be,len(edges),rp,ri,chunk=3)
        validation=exact_validate(rp,ri,bp,bi,base.arrays['group_members'],chunk=3)
        broken=ri.copy();broken[0]=(broken[0]+1)%n
        try:exact_validate(rp,broken,bp,bi,base.arrays['group_members'],chunk=3)
        except RuntimeError:pass
        else:raise RuntimeError('Corrupted overlay accepted')
        bundle=apply(base,rp,ri,g.num_edges())
        from candidates.pa_sage_bidir_native_v2.graph_io import load_pinned_csc
        for name,array in zip(('indptr','indices','eids'),(bp,bi,be)):np.save(Path(tmp)/('original_'+name+'.npy'),array)
        g,private_arrays=load_pinned_csc(Path(tmp),n,g.num_edges());records=[]
        for mode in ('gpu','gpu_i32_uva_eid64'):
            r.seed(0);sampler=DiGiTNeighborSampler([4,3,2],bundle,cuda_mode='required',metadata_mode=mode,random_seed=0)
            model=r.SAGE(128,128,172,num_layers=3,dropout=.2).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0);r.seed(0)
            steps=[]
            for _ in range(3):
                inp,out,blocks=sampler.sample_blocks(g,torch.tensor([0,2,4,6],device='cuda'));verify_sample(blocks,edges,n,'bidirectional')
                rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy();require(np.array_equal(base.arrays['storage_to_node'][rows],inp.cpu().numpy()),'Bad feature mapping')
                x=torch.from_numpy(features[inp.cpu().numpy()]).cuda();pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,(out%172));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                steps.append(dict(inp=r.digest(inp),out=r.digest(out),eids=[r.digest(b.edata[dgl.EID]) for b in blocks],rows=r.digest(blocks[0].srcdata[DIGIT_STORAGE_ROW]),flags=r.digest(blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]),loss=float(loss),model=r.model_hash(model)))
            metadata=sampler._cuda_metadata['cuda:0']
            if mode=='gpu_i32_uva_eid64':
                require(all(metadata[k].device.type=='cpu' and metadata[k].is_pinned() for k in ('original_indptr','original_indices','original_eids')),'Original CSC copied to GPU')
                require(metadata['reorganized_indices'].dtype==torch.int32 and metadata['reorganized_indptr'].dtype==torch.int64,'Wrong compact dtypes')
            records.append(steps)
        require(records[0]==records[1],'UVA64 changes sampling or training vs all-int64 CUDA')
        # Force legal int64 EID values above 2**31 in the low-level resolver.
        # The small graph avoids allocating billions of fixture edges.
        m=metadata;high=(torch.from_numpy(be.copy())+2**31).pin_memory();seeds=torch.tensor([0,1,2,3],device='cuda');fanout=4;count=len(seeds)*fanout
        outputs=[torch.empty(count,dtype=d,device='cuda') for d in (torch.int64,torch.int64,torch.uint8,torch.int64)]
        groups=torch.empty(len(seeds),dtype=torch.int64,device='cuda');nodes=torch.empty_like(groups)
        cuda.sample_group_aware_i32_uva64(*[m[k].data_ptr() for k in ('reorganized_indptr','reorganized_indices','group_members','group_storage_base','supernode_to_group','node_to_primary','original_indptr','original_indices')],high.data_ptr(),seeds.data_ptr(),len(seeds),n,base.num_groups,2,fanout,7,*[o.data_ptr() for o in outputs],groups.data_ptr(),nodes.data_ptr(),torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize();sources=outputs[0].cpu().numpy();eids=outputs[3].cpu().numpy();require((eids[sources>=0]>=2**31).all(),'EID overflow')
        for k,u in enumerate(sources):
            if u<0:continue
            owner=k//fanout;positions=np.flatnonzero(bi[bp[owner]:bp[owner+1]]==u);expected=int(be[int(bp[owner])+int(positions[0])])+2**31
            require(int(eids[k])==expected,'Wrong high EID from UVA')
        g._graph.unpin_memory_()
    result=dict(passed=True,overlay=validation,corrupt_overlay_rejected=True,three_update_training_parity=True,
                original_csc_pinned_cpu=True,eid_above_signed_int32_preserved=True,raw_ssd_access=False,
                native_sha256=sha(HERE/'runtime/digit/DiGiTSamplerCUDA.so'))
    write(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
