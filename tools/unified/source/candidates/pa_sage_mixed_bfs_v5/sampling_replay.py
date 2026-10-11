"""Replay eight real BFS smoke batches without native feature or SSD access."""
import random
from .common import *


def main():
    setup();source=verify();binding=read(OUT/'inputs.json');check_inputs(binding)
    import numpy as np,torch,dgl
    from .graph_io import load_graph,load_bundle
    from .order import prepare_orders
    from .sampling_audit import verify_digit_blocks
    from candidates.pa_sage_group_incremental_v1.sampler import IncrementalGroupSampler
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks as old_check
    from candidates.pa_sage_mean_fused_v1.model import MeanGraphSAGE,prepare_model
    from candidates.pa_sage_cpu_affinity_v1.affinity import apply
    from ae.pa_sage.common import splits
    from ae.common import host
    from GIDS import GIDS_DGLDataLoader
    p=read(HERE/'protocol.json');torch.set_num_threads(16)
    require(torch.cuda.mem_get_info()[0]>20*2**30,'GPU occupied')
    require(host()>192*2**30,'Host admission failed')
    original=read(ROOT/'results/pa_sage_cardinality_20261010_v1/summary.json')
    print('Loading anonymous graph for sampling-only replay',flush=True)
    graph,arrays=load_graph(binding,p['nodes'],p['edges']);bundle=load_bundle(binding)
    sampler=IncrementalGroupSampler(p['fanouts'],bundle,cuda_mode='required',metadata_mode=p['digit_metadata_mode'],random_seed=0,group_mode='incremental')
    sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);dgl.seed(0);dgl.random.seed(0)
    model=MeanGraphSAGE(128,128,172,3,.2,mean_mode='legacy').cuda()
    opt=torch.optim.Adam(model.parameters(),**p['optimizer'])
    prepare_model(model,opt,'graph',batch_size=1024)
    orders=prepare_orders(np.sort(splits('train')),'bfs64_ae',True,p);apply('one_core',2)
    records=[]
    for epoch,(roots,meta) in enumerate(orders):
        before=sampler._cuda_call_counter
        dl=GIDS_DGLDataLoader(graph,torch.from_numpy(roots.copy()),sampler,1024,128,None,shuffle=False,drop_last=False,num_workers=0,use_alternate_streams=False)
        for batch,(inp,out,blocks) in enumerate(iter(dl).dataloader_it):
            seed=sampler.random_seed+before+batch
            require(seed==sampler.random_seed+sampler._cuda_call_counter-1,'Unexpected sampler invocation count')
            old_passed=True
            try:old_check(blocks,arrays,p['fanouts'])
            except RuntimeError:old_passed=False
            proof=verify_digit_blocks(blocks,arrays,p['fanouts'],bundle,seed)
            rec=dict(epoch=epoch+1,batch=batch,inputs=len(inp),roots=len(out),sampler_seed=seed,old_check_passed=old_passed,new_check=proof)
            previous=original['batches'][len(records)]
            require(all(rec[k]==previous[k] for k in ('epoch','batch','inputs','roots','sampler_seed','old_check_passed')),'Replay diverged from failed-run diagnostic')
            records.append(rec);print(json.dumps(rec),flush=True)
    proofs=[proof for rec in records for proof in rec['new_check']['whole_group_no_fit']]
    require(len(records)==8 and sum(not r['old_check_passed'] for r in records)==1,'Wrong reproduction extent')
    require(len(proofs)==1 and proofs[0]['owner']==46898313 and proofs[0]['actual']==9,'Did not reproduce specific failed owner')
    require(verify()==source,'Candidate changed during replay')
    write(OUT/'sampling_replay.json',dict(passed=True,candidate_sha256=source,batches=records,sampling_only=True,raw_ssd_access=False,training_acceptance=False))
    graph._graph.unpin_memory_()
    print('PASS: all eight real batches; original false positive independently verified',flush=True)


if __name__=='__main__':main()
