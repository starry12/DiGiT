"""Real IG sampler parity, 4KiB native routing and useful-count probes; no SSD."""
import tempfile,gc
import numpy as np
from candidates.ig_perf_v5.common import *
def run_checks():
    setup()
    import torch,dgl,IGPerfNative as native
    from candidates.ig_perf_v5.tests import math_check
    from candidates.ig_perf_v5.model import make_model,optimizer
    from candidates.ig_perf_v5.worker import reset
    from sampler_config import configure
    from uva_sampler import UVANeighborSampler
    from digit.reorganization import reorganize_to_bundle
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_SAMPLED_GROUPS
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,9),'One sm89 GPU required')
    require(Path(native.__file__).resolve()==HERE/'runtime/IGPerfNative.so','Wrong native binary')
    for name,scale in (('admission_row_fixture',16),('short_raw_fixture',32)):
        want=(np.arange(3072,dtype='float32')/scale).reshape(3,1024)[[2,0,1,2,0]]
        require(np.array_equal(np.asarray(getattr(native,name)(),dtype='float32').reshape(5,1024),want),'IG CPU row mapping failed: '+name)
    # Whole-row consumption, repeated hits, region boundary warm exclusion and refill.
    events=[[1,0,0,4096],[3,0,0,4096],[4,0,0,4096],[0,0,0,4096],[3,0,0,4096],[1,0,0,4096],[3,0,0,4096],[1,1,0,4096],[3,1,0,4096]]
    counts=np.asarray(native.useful_counter_probe(events,2,4096)).reshape(-1,6)
    require(counts[:,2].tolist()==[4096,4096,4096,0,0,4096,4096,8192,8192],'IG fill counts')
    require(counts[:,3].tolist()==[0,4096,4096,0,0,0,4096,4096,8192],'IG useful unique/region counts')
    # Two 4KiB halves in an 8KiB slot: extending a fill preserves consumed mask.
    extended=[[1,0,0,4096],[3,0,0,4096],[2,0,4096,4096],[3,0,4096,4096],[3,0,0,4096]]
    values=np.asarray(native.useful_counter_probe(extended,1,8192)).reshape(-1,6)
    require(values[-1,2:4].tolist()==[8192,8192],'IG companion/extension count')
    math_results=[math_check(n,'cuda') for n in cfg()['models']];configure('full');n=16
    edges=np.array([(u,v) for v in range(n) for u in range(n) if (u+2*v)%3==0]+[(v,v) for v in range(n)]+[(0,1),(0,1)],dtype='int64')
    graph=dgl.graph((torch.from_numpy(edges[:,0].copy()),torch.from_numpy(edges[:,1].copy())),num_nodes=n).formats('csc');op,oi,oe=graph.adj_tensors('csc');graph.pin_memory_()
    features=np.random.default_rng(3).normal(size=(n,1024)).astype('float32');all_records={}
    with tempfile.TemporaryDirectory(prefix='ig-cuda-') as td:
        os.environ['DIGIT_VALIDATION_PROFILE']='legacy';bundle=reorganize_to_bundle(op.numpy(),oi.numpy(),features,td,dataset_name='IGfixture',dataset_size='tiny',group_size=2,page_size=8192,minimum_transfer_bytes=4096,target_request_bytes=8192)
        require(bundle.num_groups>0,'No groups in sampler fixture')
        for model_name in cfg()['models']:
            records=[]
            for compact in (False,True):
                reset(0);sampler=UVANeighborSampler(cfg()['fanouts'],bundle,compact=compact,random_seed=0);model=make_model(model_name,'cuda');opt=optimizer(model);reset(0);steps=[]
                for step in range(3):
                    inp,out,bs=sampler.sample_blocks(graph,torch.tensor([0,2,4,6],device='cuda'));rows=bs[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
                    require(np.array_equal(bundle.arrays['storage_to_node'][rows],inp.cpu().numpy()),'Storage/logical mismatch')
                    for block in bs:
                        u,v=block.edges();src=block.srcdata[dgl.NID][u].cpu().numpy();dst=block.dstdata[dgl.NID][v].cpu().numpy();eid=block.edata[dgl.EID].cpu().numpy()
                        require(np.array_equal(edges[eid],np.column_stack((src,dst))),'Sampled endpoint/EID differs')
                    x=torch.from_numpy(features[inp.cpu().numpy()]).cuda();pred=model(bs,x);loss=torch.nn.functional.cross_entropy(pred,out%19);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    steps.append(dict(inputs=digest(inp),eids=[digest(b.edata[dgl.EID]) for b in bs],rows=digest(bs[0].srcdata[DIGIT_STORAGE_ROW]),groups=int(bs[0].dstdata[DIGIT_SAMPLED_GROUPS].sum()),loss=float(loss),model=state_hash(model.state_dict())))
                records.append(steps);sampler.close();del sampler,model,opt,x,pred,loss,bs,inp,out;gc.collect();torch.cuda.empty_cache()
            require(records[0]==records[1],'Mixed width sampler/model parity failed: '+model_name)
            require(sum(s['groups'] for s in records[1])>0,'Grouped sampling not exercised');all_records[model_name]=records[1]
    graph._graph.unpin_memory_()
    return dict(passed=True,raw_ssd_access=False,native_sha256=sha(HERE/'runtime/IGPerfNative.so'),manual_math=math_results,ig_4k_cpu_routing=True,ig_4k_unique_fill_counts=True,grouped_sampler_parity=True,records=all_records)
if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
