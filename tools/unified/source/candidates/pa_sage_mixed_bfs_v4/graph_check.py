"""Tiny reverse-edge overlay and incremental CUDA sampler with int64 EIDs."""
import tempfile
import numpy as np
from .common import *


def run():
    import torch,dgl
    from candidates.pa_sage_direction_v2.graph import build
    from candidates.pa_sage_bidir_native_v2.overlay import rewrite,exact_validate,apply
    from digit.reorganization import reorganize_to_bundle
    from candidates.pa_sage_group_incremental_v1.sampler import IncrementalGroupSampler
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks
    from .graph_io import load_graph
    from .evaluation import model_hash
    from models import SAGE
    import DiGiTGroupIncrementalCUDA as cuda
    require(Path(cuda.__file__).resolve()==SAMPLER_BINARY,'Wrong incremental sampler')
    n=16;edges=np.array([(u,v) for v in range(n) for u in range(n) if u!=v and (u+2*v)%3==0]+[(0,1),(0,1)],dtype=np.int64)
    direct=build(edges,n,'directed');bidir=build(edges,n,'bidirectional')
    op,oi,_=[t.numpy() for t in direct.adj_tensors('csc')]
    bp,bi,be=[t.numpy() for t in bidir.adj_tensors('csc')]
    features=np.random.default_rng(3).normal(size=(n,128)).astype('float32')
    with tempfile.TemporaryDirectory(prefix='digit-ae-bidir-tiny-') as tmp:
        base=reorganize_to_bundle(op,oi,features,tmp,dataset_name='fixture',dataset_size='tiny',group_size=2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096)
        require(base.num_groups>0,'Fixture has no groups')
        rp=np.empty_like(base.arrays['reordered_indptr']);ri=np.empty(len(base.arrays['reordered_indices'])+len(edges),dtype=np.int64)
        rewrite(base.arrays['reordered_indptr'],base.arrays['reordered_indices'],op,bp,bi,be,len(edges),rp,ri,chunk=3)
        validation=exact_validate(rp,ri,bp,bi,base.arrays['group_members'],chunk=3)
        bundle=apply(base,rp,ri,len(bi))
        paths={k:str(Path(tmp)/(k+'.npy')) for k in ('indptr','indices')}
        for k,array in (('indptr',bp),('indices',bi)):np.save(paths[k],array)
        g,arrays=load_graph(dict(csc=paths),n,len(bi));records=[]
        for mode in ('legacy','incremental'):
            torch.manual_seed(0);torch.cuda.manual_seed_all(0);dgl.seed(0)
            sampler=IncrementalGroupSampler([4,3,2],bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=0,group_mode=mode)
            model=SAGE(128,128,172,3,.2).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
            steps=[]
            for _ in range(3):
                inp,out,blocks=sampler.sample(g,torch.tensor([0,2,4,6],device='cuda'))
                verify_digit_blocks(blocks,arrays,[4,3,2])
                rows=blocks[0].srcdata['digit_storage_row'].cpu().numpy()
                require(np.array_equal(base.arrays['storage_to_node'][rows],inp.cpu().numpy()),'Wrong feature inverse')
                x=torch.from_numpy(features[inp.cpu().numpy()]).cuda()
                pred=model([b.int() for b in blocks],x);loss=torch.nn.functional.cross_entropy(pred,out%172)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                steps.append(dict(inputs=inp.cpu().tolist(),eids=[b.edata[dgl.EID].cpu().tolist() for b in blocks],loss=float(loss),model=model_hash(model)))
            records.append(steps)
        require(records[0]==records[1],'Incremental algorithm changed bidirectional sampling/model')
        m=sampler._cuda_metadata['cuda:0']
        high=(torch.arange(len(bi),dtype=torch.int64)+2**31).pin_memory()
        seeds=torch.tensor([0,1,2,3],device='cuda');fanout=4;count=len(seeds)*fanout
        outputs=[torch.empty(count,dtype=d,device='cuda') for d in (torch.int64,torch.int64,torch.uint8,torch.int64)]
        groups=torch.empty(len(seeds),dtype=torch.int64,device='cuda');nodes=torch.empty_like(groups)
        cuda.sample_group_aware_i32_uva64_incremental(*[m[k].data_ptr() for k in ('reorganized_indptr','reorganized_indices','group_members','group_storage_base','supernode_to_group','node_to_primary','original_indptr','original_indices')],high.data_ptr(),seeds.data_ptr(),len(seeds),n,base.num_groups,2,fanout,7,*[o.data_ptr() for o in outputs],groups.data_ptr(),nodes.data_ptr(),torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize();sources=outputs[0].cpu().numpy();eids=outputs[3].cpu().numpy()
        for k,u in enumerate(sources):
            if u<0:continue
            owner=k//fanout;position=int(bp[owner])+int(np.flatnonzero(bi[bp[owner]:bp[owner+1]]==u)[0])
            require(int(eids[k])==position+2**31,'Truncated/wrong int64 EID')
        g._graph.unpin_memory_()
    return dict(passed=True,overlay=validation,incremental_parity=True,high_eids_preserved=True,anonymous_host_csc=True,raw_ssd_access=False)
