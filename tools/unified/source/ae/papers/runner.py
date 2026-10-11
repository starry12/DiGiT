"""Random-batch GIDS/DiGiT, persistent 20-epoch state and common full evaluation."""
from main_common import *
import argparse,random,ctypes,resource,gc
import torch,dgl
from dgl._ffi.base import _LIB
from GIDS import GIDS
from models import SAGE
from digit.artifacts import ArtifactBundle,load_artifact_bundle
from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP,DIGIT_SAMPLED_GROUPS
from digit.eval_trace import EvaluationTrace
from baseline_metrics import gpu_cache_interval,mixed_io_interval,device_io_interval,io_stats,reconcile_mixed_io_counters
PRIOR=prior()
TLS='_ZZN4dmlc16ThreadLocalStoreIN3dgl12RandomEngineEE3GetEvE4inst'

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value);dgl.seed(value)
def dgl_rng():return list((ctypes.c_uint64*2).in_dll(_LIB,TLS))
def startup():
    assert os.environ.get('USE_DETERMINISTIC_ALG')=='1'
    torch.set_num_threads(16);count=dgl.utils.get_num_threads()
    try:dgl.utils.set_num_threads(1);dgl.seed(0)
    finally:dgl.utils.set_num_threads(count)
    assert dgl_rng()[0]==1 and count==16
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def digest(value):
    a=value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()
def model_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def setup_model(value):
    seed(value);model=SAGE(128,128,172,num_layers=3,dropout=.2).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.001,betas=(.9,.999),eps=1e-8)
    initial=model_hash(model);seed(value);model.train();return model,optimizer,initial

class SourceLoader:
    is_source=True
    def __init__(self,features):self.features=features;self.rows=0;self.sample_time=0.;self.feature_time=0.
    def fetch_feature(self,dim,it,device):
        t=time.perf_counter();inp,out,blocks=next(it);self.sample_time+=time.perf_counter()-t
        t=time.perf_counter();x=torch.from_numpy(np.ascontiguousarray(self.features[inp.cpu().numpy()])).cuda()
        self.feature_time+=time.perf_counter()-t;self.rows+=len(inp);return inp,out,blocks,x

def snapshot(loader):
    torch.cuda.synchronize()
    if isinstance(loader,SourceLoader):return dict(source_rows=loader.rows,sampling_seconds=loader.sample_time,feature_seconds=loader.feature_time)
    return dict(gpu=loader.get_gpu_cache_stats(),mixed=loader.get_mixed_io_stats(),device=loader.get_device_io_stats(),
                feature=loader.get_feature_access_stats(),native=loader.get_io_stat(),sampling_seconds=loader.sample_time,feature_seconds=loader.feature_time)
def interval(a,b,rows):
    if 'source_rows' in a:
        assert b['source_rows']-a['source_rows']==rows
        return dict(source_only=True,rows=rows,sampling_seconds=b['sampling_seconds']-a['sampling_seconds'],feature_seconds=b['feature_seconds']-a['feature_seconds'])
    gpu=gpu_cache_interval(a['gpu'],b['gpu']);mixed=mixed_io_interval(a['mixed'],b['mixed'])
    dev=device_io_interval(a['device'],b['device'],maxima_scope='epoch_cumulative')
    dev['maxima_scope']='run_cumulative_since_last_reset'
    for key in ('max_latency_ns','max_outstanding'):
        if 'epoch_'+key in dev:dev['run_'+key]=dev.pop('epoch_'+key)
    native=io_stats(b['native']-a['native']);feature={k:b['feature'][k]-a['feature'][k] for k in b['feature']}
    assert sum(feature.values())==rows and min(feature.values())>=0 and gpu['requests']==feature['gpu_ssd']
    return dict(gpu=gpu,mixed=mixed,device=dev,native=native,feature=feature,
        sampling_seconds=b['sampling_seconds']-a['sampling_seconds'],feature_seconds=b['feature_seconds']-a['feature_seconds'],
        reconciliation=reconcile_mixed_io_counters(native,gpu,mixed,dev))

def initialize(arm,source=False):
    from digit import ssd_payload as sp
    from digit.gpu_admission import check_live
    admission=check_live(read(D/'admission.json'));assert admission['passed'],admission
    progress(stage='admitted',arm=arm,free_gpu=admission['free_bytes'],host_available=admission['host_available_bytes'])
    bundle=load_artifact_bundle(S/'bundle',validation_mode='fast') if arm=='digit_full' else None
    cpu=arr(S/('full_cpu_rows.npy' if bundle else 'gids_cpu_rows.npy'))
    assert len(cpu)==PRIOR.K and np.all(np.diff(cpu)>0)
    if source:return None,bundle,cpu,admission
    if bundle:
        sp.validate_bundle_verify_receipt(S/'ssd/full_verify.json',bundle,device_offset_bytes=320*2**30)
        offset=320*2**30;num_ele=bundle.manifest['feature']['num_storage_rows']*128
    else:
        plan=sp.build_plain_payload_plan(read(BASE_RECEIPT)['feature_file'],device_offset_bytes=128*2**30)
        sp.validate_verify_receipt(BASE_RECEIPT,plan);offset=plan.device_offset_bytes;num_ele=plan.storage_rows*128
    loader=GIDS(page_size=4096,off=offset,cache_dim=128,num_ele=num_ele,num_ssd=1,ssd_list=[0],cache_size=512,
        ctrl_idx=0,window_buffer=False,accumulator_flag=False,feature_index_mode='explicit' if bundle else 'logical',
        gpu_cache_policy='fifo' if bundle else 'legacy',cpu_feature_path='mapped',mixed_io=bool(bundle),
        mixed_io_geometry=bundle.io_geometry.with_payload_offset(offset).to_native_mapping() if bundle else None,device_io_stats=True)
    t=time.perf_counter();loader.cpu_backing_buffer(128,PRIOR.K);loader.set_cpu_buffer(torch.from_numpy(cpu.copy()),PRIOR.K)
    torch.cuda.synchronize();loader.reset_device_io_stats()
    progress(stage='cpu_preloaded',arm=arm,seconds=time.perf_counter()-t)
    assert loader.get_gpu_cache_stats()['capacity_pages']==131072
    return loader,bundle,cpu,admission

def batch_audit(inp,out,blocks,x,features,bundle):
    logical=inp.cpu().numpy();got=x.cpu().numpy();want=features[logical]
    assert np.array_equal(got.view(np.uint32),want.view(np.uint32)) and np.isfinite(got).all()
    if bundle:assert np.array_equal(bundle.arrays['storage_to_node'][blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()],logical)
    rec=dict(feature=digest(got),inputs=digest(inp),outputs=digest(out),cuda_rng=digest(torch.cuda.get_rng_state()),dgl_rng=dgl_rng(),blocks=[])
    for b in blocks:
        u,v=b.edges(order='eid');rec['blocks'].append(dict(src=digest(b.srcdata[dgl.NID]),dst=digest(b.dstdata[dgl.NID]),u=digest(u),v=digest(v),eids=digest(b.edata[dgl.EID])))
    if bundle:rec.update(storage_rows=digest(blocks[0].srcdata[DIGIT_STORAGE_ROW]),storage_flags=digest(blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]))
    return rec

def evaluate(model,loader,bundle,labels,trace,limit=None,audit_features=None):
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all(),dgl_rng())
    model.eval();before=snapshot(loader);began=time.perf_counter();guesses=[];targets=[];losses=[];rows=0;bindings=[]
    with torch.no_grad():
        for i,item in enumerate(trace.iter_batches(device='cuda:0')):
            if limit is not None and i>=limit:break
            inp,out,blocks=item
            if bundle:
                physical=bundle.arrays['node_to_primary_row'][inp.cpu().numpy()].copy()
                blocks[0].srcdata[DIGIT_STORAGE_ROW]=torch.from_numpy(physical).cuda()
                blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]=torch.zeros(len(inp),dtype=torch.bool,device='cuda')
            _,_,_,x=loader.fetch_feature(128,iter([item]),torch.device('cuda:0'))
            y=torch.from_numpy(labels[out.cpu().numpy()].astype(np.int64)).cuda();pred=model(blocks,x)
            if audit_features is not None:
                assert torch.isfinite(pred).all()
                assert np.array_equal(x.cpu().numpy().view(np.uint32),audit_features[inp.cpu().numpy()].view(np.uint32))
                bindings.append(dict(input=digest(inp),output=digest(out),feature=digest(x),labels=digest(y)))
            guesses.append(pred.argmax(1));targets.append(y);losses.append(torch.nn.functional.cross_entropy(pred,y,reduction='sum'));rows+=len(inp)
    torch.cuda.synchronize();seconds=time.perf_counter()-began;stats=interval(before,snapshot(loader),rows)
    pred=torch.cat(guesses).cpu().numpy();target=torch.cat(targets).cpu().numpy();hit=pred==target
    assert dgl_rng()==states[4],'Evaluation changed DGL RNG'
    random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2]);torch.cuda.set_rng_state_all(states[3]);model.train()
    assert digest(torch.cuda.get_rng_state())==digest(states[3][0])
    return dict(seconds=seconds,accuracy=float(hit.mean()),loss=float(torch.stack(losses).sum())/len(target),examples=len(target),batches=len(guesses),
        prediction_sha256=digest(pred),prediction_histogram=np.bincount(pred,minlength=172).tolist(),
        per_class_correct=np.bincount(target[hit],minlength=172).tolist(),bindings=bindings,trace_manifest_sha256=trace.manifest_sha256,
        training_rng_restored=True,**stats)

def run_training(arm,value,loader,bundle,graph,labels,degrees,output,smoke=False,features=None):
    sampler=(DiGiTNeighborSampler([16,5,5],bundle,cuda_mode='required',metadata_mode='gpu_i32',random_seed=value)
             if bundle else dgl.dataloading.NeighborSampler([16,5,5],replace=False))
    train_ids=PRIOR.splits('train');order_hashes=read(D/'orders.json')
    valid=EvaluationTrace(S/'valid_trace');valid.validate_contract(PRIOR.N,PRIOR.E,[16,5,5],1024,PRIOR.splits('valid'))
    assert valid.manifest_sha256==read(D/'inputs_ready.json')['traces']['valid']['manifest_sha256']
    model,optimizer,initial=setup_model(value);initial_rng=dgl_rng();epochs=[];audits=[];metadata=None
    identities=(id(graph),id(sampler),id(loader),id(model),id(optimizer));start_cache=snapshot(loader)
    for epoch in range(20):
        assert identities==(id(graph),id(sampler),id(loader),id(model),id(optimizer))
        begin=snapshot(loader);previous=begin;epoch_start=time.perf_counter()
        roots=np.random.default_rng(np.random.SeedSequence([value,epoch])).permutation(train_ids)
        root_hash=digest(roots);assert root_hash==order_hashes['s%d_e%d'%(value,epoch)]
        root_gpu=torch.from_numpy(roots.copy()).cuda();ys=torch.from_numpy(labels[roots].astype(np.int64)).cuda()
        limit=2 if smoke else 1179
        def batches():
            for i in range(limit):yield sampler.sample_blocks(graph,root_gpu[i*1024:(i+1)*1024])
        iterator=batches();torch.cuda.synchronize();order_seconds=time.perf_counter()-epoch_start
        window_start=time.perf_counter();losses=[];coverage=[];shapes=[];windows=[];window_rows=0;total_rows=0
        for i in range(limit):
            inp,out,blocks,x=loader.fetch_feature(128,iterator,torch.device('cuda:0'))
            record=batch_audit(inp,out,blocks,x,features,bundle) if smoke else None
            if smoke:assert np.array_equal(out.cpu().numpy(),roots[i*1024:(i+1)*1024])
            pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,ys[i*1024:i*1024+len(out)])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if smoke:assert torch.isfinite(pred).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();losses.append(loss.detach())
            target=degrees[blocks[0].dstdata[dgl.NID]].to(torch.int64).clamp(max=16);actual=blocks[0].in_degrees().to(torch.int64)
            groups=blocks[0].dstdata[DIGIT_SAMPLED_GROUPS].sum()*8 if bundle else target.new_zeros(())
            coverage.append(torch.stack([target.sum(),actual.sum(),(actual<target).sum(),(target-actual).sum(),groups,(actual>target).sum()]))
            shapes.append(dict(input_nodes=len(inp),output_nodes=len(out),block_edges=[b.num_edges() for b in blocks]))
            total_rows+=len(inp);window_rows+=len(inp)
            if smoke:
                record.update(epoch=epoch,batch=i,loss=float(loss),parameters_after=model_hash(model));audits.append(record)
            if (i+1)%100==0 or i+1==limit:
                torch.cuda.synchronize();stop=time.perf_counter();current=snapshot(loader)
                cov=torch.stack(coverage).sum(0).cpu().tolist();coverage=[];assert cov[-1]==0
                windows.append(dict(start_update=1 if not windows else windows[-1]['end_update']+1,end_update=i+1,
                    wall_seconds=stop-window_start,input_nodes=window_rows,target_edges=cov[0],actual_edges=cov[1],
                    underfilled_owners=cov[2],shortfall_edges=cov[3],group_edges=cov[4],**interval(previous,current,window_rows)))
                previous=current;window_rows=0;window_start=time.perf_counter()
                if not smoke:progress(stage='training',arm=arm,seed=value,epoch=epoch+1,updates=i+1)
        torch.cuda.synchronize();train_seconds=time.perf_counter()-epoch_start
        values=torch.stack(losses).cpu().numpy();assert np.isfinite(values).all() and all(torch.isfinite(p).all() for p in model.parameters())
        assert len(shapes)==limit and (smoke or sum(s['output_nodes'] for s in shapes)==1207179)
        assert smoke or shapes[-1]['output_nodes']==907
        if bundle:
            storage={k:v.data_ptr() for k,v in sampler._cuda_metadata['cuda:0'].items()}
            if metadata is None:metadata=storage
            assert metadata==storage and sampler._cuda_call_counter==(epoch+1)*limit
        ev=evaluate(model,loader,bundle,labels,valid,10 if smoke else None,features if smoke else None)
        assert ev['examples']==(10240 if smoke else 125265) and ev['batches']==(10 if smoke else 123)
        report=dict(epoch=epoch+1,updates=limit,root_sha256=root_hash,train_seconds=train_seconds,order_seconds=order_seconds,
                    losses=values.tolist(),loss_sha256=digest(values),shapes=shapes,windows=windows,training=interval(begin,previous,total_rows),
                    validation=ev,model_sha256=model_hash(model),metadata_reused=True,cache_object_reused=True,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        epochs.append(report);write(output.parent/('epoch_%02d.json'%(epoch+1)),report)
        progress(stage='epoch_complete',arm=arm,seed=value,epoch=epoch+1,valid_accuracy=ev['accuracy'],train_seconds=train_seconds,smoke=smoke)
    test_result=None
    if not smoke:
        test=EvaluationTrace(TEST);test.validate_contract(PRIOR.N,PRIOR.E,[16,5,5],1024,PRIOR.splits('test'))
        assert test.manifest_sha256==read(D/'inputs_ready.json')['traces']['test']['manifest_sha256']
        test_result=evaluate(model,loader,bundle,labels,test)
        assert test_result['examples']==214338 and test_result['batches']==210
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output.parent/'final_model.pt')
    end_cache=snapshot(loader)
    for point in (start_cache,end_cache):
        if 'native' in point:point['native']=point['native'].tolist()
    return dict(passed=True,arm=arm,seed=value,smoke=smoke,epochs=epochs,updates=40 if smoke else 23580,
        initial_parameters_sha256=initial,initial_dgl_rng=initial_rng,final_parameters_sha256=model_hash(model),
        all_epochs_metadata_and_cache_reused=True,audits=audits,test=test_result,
        initial_cache=start_cache,final_cache=end_cache,training_seconds=sum(r['train_seconds'] for r in epochs),
        validation_seconds=sum(r['validation']['seconds'] for r in epochs),source_only=isinstance(loader,SourceLoader),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
