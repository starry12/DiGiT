"""Native IG grouped-sampler equivalence and independent compaction trace; no SSD."""
import gc
import tempfile
from types import SimpleNamespace
from .common import *


def run():
    import subprocess
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','GPU2 required')
    gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
    require(gpu[0].strip()=='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703' and int(gpu[1])<1024,'GPU2 busy/different')
    setup()
    import numpy as np
    import torch,dgl
    from sampler_config import configure
    from digit.reorganization import reorganize_to_bundle
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP,DIGIT_SAMPLED_GROUPS
    from .sampler import sampler_type,compact_columns
    from .model import make_model,optimizer
    from .worker import reset
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configure('full');n=32
    edges=np.array([(u,v) for v in range(n) for u in range(n) if (u+2*v)%3==0]+[(v,v) for v in range(n)]+[(0,1),(0,1)],dtype='int64')
    graph=dgl.graph((torch.from_numpy(edges[:,0].copy()),torch.from_numpy(edges[:,1].copy())),num_nodes=n).formats('csc');op,oi,oe=graph.adj_tensors('csc');graph.pin_memory_()
    features=np.random.RandomState(3).normal(size=(n,1024)).astype('float32');all_records={}
    with tempfile.TemporaryDirectory(prefix='ig-shared-index-') as td:
        os.environ['DIGIT_VALIDATION_PROFILE']='legacy'
        bundle=reorganize_to_bundle(op.numpy(),oi.numpy(),features,td,dataset_name='IGfixture',dataset_size='tiny',group_size=2,page_size=8192,minimum_transfer_bytes=4096,target_request_bytes=8192)
        require(bundle.num_groups>0,'Fixture must exercise group edges')
        for variant in ('legacy','compact'):
            reset(0);sampler=sampler_type(variant)(cfg()['fanouts'],bundle,random_seed=0);model=make_model('sage','cuda');opt=optimizer(model);reset(0);steps=[]
            for step in range(12):
                roots=torch.tensor([step%8,(step+2)%16,20,30],device='cuda')
                inp,out,bs=sampler.sample_blocks(graph,roots);rows=bs[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
                require(np.array_equal(bundle.arrays['storage_to_node'][rows],inp.cpu().numpy()),'Storage/logical mismatch')
                block_hashes=[]
                for block in bs:
                    u,v=block.edges();src=block.srcdata[dgl.NID][u].cpu().numpy();dst=block.dstdata[dgl.NID][v].cpu().numpy();eid=block.edata[dgl.EID].cpu().numpy()
                    require(np.array_equal(edges[eid],np.column_stack((src,dst))),'Sampled endpoint/EID differs')
                    block_hashes.append(dict(coo=digest(torch.stack([u,v])),eids=digest(eid),src=digest(block.srcdata[dgl.NID]),dst=digest(block.dstdata[dgl.NID])))
                x=torch.from_numpy(features[inp.cpu().numpy()]).cuda();pred=model(bs,x);loss=torch.nn.functional.cross_entropy(pred,out%19)
                opt.zero_grad(set_to_none=True);loss.backward()
                gradients=state_hash({k:v.grad for k,v in model.named_parameters()});opt.step()
                steps.append(dict(inputs=digest(inp),blocks=block_hashes,rows=digest(bs[0].srcdata[DIGIT_STORAGE_ROW]),flags=digest(bs[0].srcdata[DIGIT_STORAGE_IS_GROUP]),
                    group_edges=int(bs[0].dstdata[DIGIT_SAMPLED_GROUPS].sum()),logits=digest(pred),loss=float(loss),gradients=gradients,model=state_hash(model.state_dict()),adam=state_hash(opt.state_dict())))
            all_records[variant]=steps;sampler.close();del sampler,model,opt,x,pred,loss,bs,inp,out;gc.collect();torch.cuda.empty_cache()
    require(all_records['legacy']==all_records['compact'],'Compaction changed native block/model/Adam results')
    require(sum(r['group_edges'] for r in all_records['compact'])>0,'Grouped requests not exercised')
    graph._graph.unpin_memory_()
    # Independent synthetic native-output trace. This is not E2E timing.
    size=16000;fanout=10;count=size*fanout
    sources=torch.arange(count,device='cuda',dtype=torch.int64)%269346174;sources[::11]=-1
    rows=torch.arange(count,device='cuda',dtype=torch.int64);flags=(rows%3==0).to(torch.uint8);eids=rows+2**33;seeds=torch.arange(size,device='cuda',dtype=torch.int64)
    def legacy():
        valid=sources>=0
        return sources[valid],seeds.repeat_interleave(fanout)[valid],rows[valid],flags[valid],eids[valid]
    def compact():return compact_columns(sources,rows,flags,eids,seeds,fanout)
    require(all(torch.equal(a,b) for a,b in zip(legacy(),compact())),'Large compaction output mismatch')
    traces={}
    for name,fn in [('legacy',legacy),('compact',compact)]:
        for _ in range(5):fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(4):fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(OUT/(name+'_compaction_trace.json')))
        events={event.key:dict(count=event.count,cpu_us=event.cpu_time_total,cuda_us=event.cuda_time_total) for event in prof.key_averages()}
        traces[name]=events
    require(traces['legacy']['aten::nonzero']['count']==20 and traces['compact']['aten::nonzero']['count']==4,'Unexpected repeated valid-index scan count')
    result=dict(passed=True,raw_ssd_access=False,native_sampler_changed=False,updates_per_variant=12,all_native_blocks_logits_gradients_loss_parameters_adam_exact=True,
        native_group_records=all_records,compaction_trace_events=traces,nonzero_per_frontier=dict(legacy=5,compact=1),
        trace_scope='Four synthetic 16000-root frontier compactions; not an IG full-graph/E2E performance claim')
    write(OUT/'gpu_checks.json',result)
    print(json.dumps(dict(passed=True,updates_per_variant=12,nonzero_per_frontier=result['nonzero_per_frontier']),indent=2))


if __name__=='__main__':run()
