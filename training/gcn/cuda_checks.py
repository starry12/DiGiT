"""Tiny actual CUDA sampler plus GCN updates; no raw SSD or PA-sized allocation."""
import tempfile
import numpy as np
from training.gcn.common import *

def run_checks():
    setup()
    import torch,dgl,runner as r
    from training.gcn.model import make_model
    from training.gcn.tests import math_check
    from ae.papers.graph_checks import build,verify_sample
    from training.sage.overlay import rewrite,apply,exact_validate
    from training.sage.graph_io import load_pinned_csc
    from digit.reorganization import reorganize_to_bundle
    from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW,DIGIT_SAMPLED_GROUPS
    from digit import DiGiTSamplerCUDA
    require(Path(DiGiTSamplerCUDA.__file__).resolve()==PARENT/'runtime/digit/DiGiTSamplerCUDA.so','Wrong frozen sampler')
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,9),'Expected one visible sm89 GPU')
    r.startup();math_result=math_check('cuda');n=16
    edges=np.array([(u,v) for v in range(n) for u in range(n) if u!=v and (u+2*v)%3==0]+[(0,1),(0,1)],dtype=np.int64)
    direct=build(edges,n,'directed');bidir=build(edges,n,'bidirectional')
    op,oi,_=[t.numpy() for t in direct.adj_tensors('csc')];bp,bi,be=[t.numpy() for t in bidir.adj_tensors('csc')]
    features=np.random.default_rng(3).normal(size=(n,128)).astype('float32');p=cfg();records=[]
    with tempfile.TemporaryDirectory(prefix='digit-gcn-cuda-') as td:
        base=reorganize_to_bundle(op,oi,features,td,dataset_name='fixture',dataset_size='tiny',group_size=2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096)
        require(base.num_groups>0,'No fixture groups')
        rp=np.empty_like(base.arrays['reordered_indptr']);ri=np.empty(len(base.arrays['reordered_indices'])+len(edges),dtype=np.int64)
        rewrite(base.arrays['reordered_indptr'],base.arrays['reordered_indices'],op,bp,bi,be,len(edges),rp,ri,chunk=3)
        exact_validate(rp,ri,bp,bi,base.arrays['group_members'],chunk=3);bundle=apply(base,rp,ri,bidir.num_edges())
        for name,array in zip(('indptr','indices','eids'),(bp,bi,be)):np.save(Path(td)/('original_'+name+'.npy'),array)
        graph,private=load_pinned_csc(Path(td),n,bidir.num_edges())
        for mode in ('gpu','gpu_i32_uva_eid64'):
            r.seed(0);sampler=DiGiTNeighborSampler(p['fanouts'],bundle,cuda_mode='required',metadata_mode=mode,random_seed=0);model=make_model('cuda')
            options=dict(p['optimizer']['kwargs']);options['betas']=tuple(options['betas']);optimizer=torch.optim.Adam(model.parameters(),**options);r.seed(0);steps=[]
            for step in range(3):
                inp,out,bs=sampler.sample_blocks(graph,torch.tensor([0,2,4,6],device='cuda'));verify_sample(bs,edges,n,'bidirectional')
                rows=bs[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy();require(np.array_equal(base.arrays['storage_to_node'][rows],inp.cpu().numpy()),'Feature mapping differs')
                x=torch.from_numpy(features[inp.cpu().numpy()]).cuda();pred=model(bs,x);loss=torch.nn.functional.cross_entropy(pred,out%172)
                optimizer.zero_grad(set_to_none=True);loss.backward();require(all(torch.isfinite(q.grad).all() for q in model.parameters()),'Nonfinite GCN gradient');optimizer.step()
                steps.append(dict(inputs=r.digest(inp),outputs=r.digest(out),eids=[r.digest(b.edata[dgl.EID]) for b in bs],rows=r.digest(bs[0].srcdata[DIGIT_STORAGE_ROW]),groups=int(bs[0].dstdata[DIGIT_SAMPLED_GROUPS].sum()),loss=float(loss),model=r.model_hash(model)))
            records.append(steps)
        require(records[0]==records[1],'GCN training differs between int64 CUDA and mixed-width UVA sampler')
        require(sum(s['groups'] for s in records[1])>0,'GCN fixture did not exercise grouped sampling')
        graph._graph.unpin_memory_()
    torch.cuda.synchronize()
    return dict(passed=True,manual_gcn_math=math_result,updates_per_backend=3,fanouts=p['fanouts'],g2_grouped_sampling=True,
                bit_exact_sampling_loss_and_parameter_parity=True,model_parameters=p['model_parameter_count'],raw_ssd_access=False,
                sampler_sha256=sha(PARENT/'runtime/digit/DiGiTSamplerCUDA.so'),observed_torch_peak_bytes=torch.cuda.max_memory_allocated())
if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
