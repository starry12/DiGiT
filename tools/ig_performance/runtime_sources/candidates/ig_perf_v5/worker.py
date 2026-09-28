"""Bounded IG training windows; no evaluation or accuracy path."""
import argparse,fcntl,math,random,gc,resource
import numpy as np
from candidates.ig_perf_v5.common import *

def reset(seed):
    import torch,dgl
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);dgl.seed(seed)

def execute(a):
    execution=verify();p=cfg();binding=read(a.binding);check_inputs(binding)
    require(binding['candidate_sha256']==execution and binding['protocol_sha256']==sha(P),'Wrong input/code binding')
    setup()
    import torch,dgl
    import IGPerfNative as native
    from sampler_config import configure
    from bounded_io import load_csc
    from uva_sampler import UVANeighborSampler
    from digit.artifacts import ArtifactBundle
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP,DIGIT_SAMPLED_GROUPS
    from candidates.ig_perf_v5.model import make_model,model_config,optimizer
    from candidates.ig_perf_v5.features import FormalFeatures,aggregate
    from candidates.ig_perf_v5.admission import check_live
    from candidates.io_accounting_v1.accounting import summarize
    from ae.pa_sage.observations import CheckpointObservations
    from ae.common import check_device
    require(Path(native.__file__).resolve()==HERE/'runtime/IGPerfNative.so','Wrong useful I/O binary')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,9),'Expected one visible sm89 GPU')
    plan=check_live();require(plan['passed'] and plan['free_bytes']>plan['total_bytes']-2**30,'GPU/host budget failed or GPU occupied')
    torch.cuda.set_per_process_memory_fraction(p['torch_allocator_cap_bytes']/plan['total_bytes'])
    check_device();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();observer=CheckpointObservations(a.output/'resources.json')
    def mark(stage,**kw):
        observer.mark(stage);free,total=torch.cuda.mem_get_info()
        require(free>=p['runtime_gpu_floor_bytes'] and total-free<=plan['required_bytes'] and host()>=p['runtime_host_floor_bytes'],'Runtime memory floor/budget failed')
        progress(a.output,stage,arm=a.arm,model=a.model,smoke=a.smoke,**kw)
    mark('start');arm='full' if a.arm=='digit_full' else 'gids';sampler_configuration=configure(arm)
    features=FormalFeatures(arm,a.output,mark)
    cm=read(DATA/'csc/manifest.json');names=['original_indptr.npy','original_indices.npy','original_eids.npy']
    graph,csc=load_csc(*[DATA/'csc'/name for name in names],p['nodes'],host_cap=70*2**30,expected_sha256=[cm['files'][name]['payload_sha256'] for name in names])
    require(graph.num_edges()==p['edges'],'Wrong IG graph');mark('csc_pinned');arrays=metadata=bundle=None
    if arm=='full':
        manifest=read(DATA/'full/manifest.json');arrays={k:np.load(DATA/'full'/v['path'],mmap_mode='r') for k,v in manifest['files'].items() if k!='reordered_features'}
        bundle=ArtifactBundle(DATA/'full',manifest,arrays);sampler=UVANeighborSampler(p['fanouts'],bundle,random_seed=0,host_cap=100*2**30)
        metadata=sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
        require(metadata.accounting['original_csc_owned_bytes']==0 and metadata.accounting['host_logical_bytes']==plan['host_metadata_total_bytes'],'IG metadata ownership/budget differs')
        write(a.output/'metadata.json',metadata.accounting);mark('metadata_pinned')
    else:sampler=dgl.dataloading.NeighborSampler(p['fanouts'],replace=False)
    features.populate(None);mark('cpu_cache_ready')
    source=read(ROOT/'configs/igb/dataset.json')['dataset'];labels=np.memmap(source['labels']['path'],dtype='<f4',mode='r',shape=(p['nodes'],))
    raw=np.memmap(source['feature']['path'],dtype='<f4',mode='r',shape=(p['nodes'],1024)) if a.smoke else None
    audits=[]
    if a.smoke:
        class Raw:
            def rows(self,ids):return np.asarray(raw[ids])
        features.store.begin_useful_io_region();features.routing(Raw(),arrays)
        mark('routing_verified')
    reset(0);model=make_model(a.model,'cuda');opt=optimizer(model);initial=state_hash(model.state_dict());reset(0)
    original_ptrs=[x.data_ptr() for x in graph.adj_tensors('csc')]
    owners=(id(graph),id(sampler),id(metadata),id(model),id(opt),id(features))
    metadata_ptrs={k:v.data_ptr() for k,v in metadata.items()} if metadata else {}
    def retained():
        require(owners==(id(graph),id(sampler),id(metadata),id(model),id(opt),id(features)),'Objects recreated')
        require(graph.is_pinned() and original_ptrs==[x.data_ptr() for x in graph.adj_tensors('csc')],'CSC moved/unpinned')
        if metadata:require(metadata_ptrs=={k:v.data_ptr() for k,v in metadata.items()},'Metadata pointers changed')
    def fetch(inputs,blocks,standard):
        logical=inputs.cpu().numpy()
        physical=np.asarray(arrays['node_to_primary_row'][logical],dtype=np.int64) if arrays is not None and standard else blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy() if arrays is not None else logical
        flags=blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP].cpu().numpy() if arrays is not None and not standard else np.zeros(len(logical),dtype=np.bool_)
        x=torch.empty((len(logical),1024),device='cuda')
        for at in range(0,len(logical),16384):x[at:at+16384].copy_(features.fetch(physical[at:at+16384],flags[at:at+16384]))
        if a.smoke:
            require(np.array_equal(x.cpu().numpy().view('u4'),np.asarray(raw[logical]).view('u4')),'Native/source feature mismatch')
            # Full-block CSC membership for the bounded smoke; original EIDs carry graph provenance.
            ip,ix,_=graph.adj_tensors('csc')
            for block in blocks:
                u,v=block.edges();src=block.srcdata[dgl.NID][u].cpu().numpy();dst=block.dstdata[dgl.NID][v].cpu().numpy()
                for node in np.unique(dst):
                    neighbors=ix[int(ip[node]):int(ip[node+1])].numpy();require(np.isin(src[dst==node],neighbors).all(),'Sampled edge absent from IG CSC')
            audits.append(dict(rows=len(logical),source_equal=True,features_sha256=digest(x),standard=standard))
        return x
    def phase(name,roots,phase_sampler,width):
        retained();model.train();features.store.begin_useful_io_region();start_region=int(features.store.get_useful_io_stats()[1])
        sampler_start=getattr(sampler,'_cuda_call_counter',0)
        torch.cuda.synchronize();began=time.perf_counter();windows=[];pending=[];pending_groups=[];loss_values=[];rows_seen=window_rows=examples=group_edges=outer_edges=0
        root_hash=hashlib.sha256();shape_totals=np.zeros((3,3),dtype=np.int64);batches=len(roots)//p['batch_size']
        require(len(roots)%p['batch_size']==0,'Partial batches are not permitted')
        for i in range(batches):
            if i%width==0:
                before=features.begin();window_rows=0;window_root_hash=hashlib.sha256();edge_before=outer_edges
                torch.cuda.synchronize();window_started=time.perf_counter()
            rows=np.asarray(roots[i*p['batch_size']:(i+1)*p['batch_size']],dtype=np.int64)
            root=torch.from_numpy(rows.copy()).cuda();inputs,outputs,blocks=phase_sampler.sample_blocks(graph,root)
            require(len(outputs)==len(rows),'Wrong output extent');x=fetch(inputs,blocks,False)
            if a.smoke:require(np.array_equal(outputs.cpu().numpy(),rows),'Root order changed')
            targets=np.asarray(labels[rows]);require(np.isfinite(targets).all() and np.all(targets==targets.astype('int64')) and targets.min()>=0 and targets.max()<19,'Invalid supervised labels')
            y=torch.from_numpy(targets.astype('int64')).cuda();pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,y)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            pending.append(loss.detach());examples+=len(rows);root_hash.update(rows.tobytes());window_root_hash.update(rows.tobytes());rows_seen+=len(inputs);window_rows+=len(inputs)
            shape_totals+=np.array([[b.num_src_nodes(),b.num_dst_nodes(),b.num_edges()] for b in blocks]);outer_edges+=blocks[0].num_edges()
            if arrays is not None:pending_groups.append(blocks[0].dstdata[DIGIT_SAMPLED_GROUPS].sum())
            if (i+1)%width==0 or i+1==batches:
                torch.cuda.synchronize();window_seconds=time.perf_counter()-window_started
                losses=torch.stack(pending).cpu().numpy();require(np.isfinite(losses).all(),'Nonfinite losses');loss_values.extend(map(float,losses));pending=[]
                window_groups=int(torch.stack(pending_groups).sum())*p['group_size'] if pending_groups else 0
                group_edges+=window_groups;pending_groups=[]
                record=features.finish(before,window_rows)
                record.update(index=len(windows),batch_start=i-i%width,batch_end=i+1,batches=i%width+1,seconds=window_seconds,roots_sha256=window_root_hash.hexdigest(),group_edges=window_groups,outer_edges=outer_edges-edge_before)
                windows.append(record);append_sync(a.output/'io_windows.jsonl',dict(phase=name,**record))
                progress(a.output,name,arm=a.arm,model=a.model,updates=i+1,total_updates=batches,examples=examples,seconds=sum(w['seconds'] for w in windows))
                free,total=torch.cuda.mem_get_info();require(free>=p['runtime_gpu_floor_bytes'] and total-free<=plan['required_bytes'],'Runtime GPU budget failed')
        torch.cuda.synchronize();wall_seconds=time.perf_counter()-began;seconds=sum(w['seconds'] for w in windows);region=aggregate(windows)
        if arrays is not None:require(sampler._cuda_call_counter-sampler_start==batches,'Wrong grouped sampler call count')
        require(region['useful_io']['region_id']==start_region and sum(region['feature'].values())==rows_seen,'Phase I/O extent differs')
        require(all(torch.isfinite(v).all() for v in model.parameters()),'Nonfinite model')
        result=dict(name=name,examples=examples,batches=batches,seconds=seconds,phase_wall_seconds=wall_seconds,roots_sha256=root_hash.hexdigest(),losses=loss_values,sampling_shape_totals=shape_totals.tolist(),feature_rows=rows_seen,
                    group_edges=group_edges,outer_edges=outer_edges,window_count=len(windows),windows=windows,**region)
        write(a.output/(name+'.json'),result);retained();return result
    from candidates.ig_perf_v5.windows import root_slices
    order=np.load(ROOT/p['train_order'],mmap_mode='r');ranges=root_slices(p,a.smoke)
    require(order.shape==(p['total_train_batches']*p['batch_size'],),'Wrong frozen root count')
    mark('training_ready');setup_seconds=time.perf_counter()-start;phases={}
    for name,start_batch,end_batch,width in ranges:
        phases[name]=phase(name,order[start_batch*p['batch_size']:end_batch*p['batch_size']],sampler,width)
    train=phases['training'];warmup=phases.get('warmup');updates=sum(v['batches'] for v in phases.values())
    mark('bounded_training_complete',updates=updates,measured_updates=0 if a.smoke else train['batches'])
    checkpoint_model=state_hash(model.state_dict())
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),updates=updates,epoch=None,model_name=a.model),a.output/'final_model.pt')
    retained();features.final_checks();check_inputs(binding);require(verify()==execution,'Code changed during training');mark('training_complete')
    report=dict(schema='digit-ig-window-report-v1',passed=True,arm=a.arm,model_name=a.model,model_config=model_config(a.model),model_parameter_count=sum(q.numel() for q in model.parameters()),optimizer=p['optimizer'],
                seed=0,epochs=None,smoke=a.smoke,updates=updates,training=train,warmup=warmup,measured_batches=0 if a.smoke else train['batches'],validation=None,validation_calls=0,test=None,test_calls=0,diagnostic_replays=0,
                epoch_time_claim=False,steady_state_proven=False,accuracy=None,final_accuracy_claim=False,
                initial_parameters_sha256=initial,final_parameters_sha256=checkpoint_model,checkpoint_sha256=sha(a.output/'final_model.pt'),candidate_sha256=execution,protocol_sha256=sha(P),input_binding_sha256=sha(a.binding),
                raw_ssd_writes=False,all_metadata_and_cache_reused=True,route_warmup=a.smoke,source_audits=audits,routing=read(a.output/'routing.json') if a.smoke else None,
                native=features.receipt(),sampler_configuration=sampler_configuration,csc=csc,admission=plan,resource_observations=observer.result(),
                io_accounting_training=summarize([(train,train['seconds'])]),io_accounting_warmup=summarize([(warmup,warmup['seconds'])]) if warmup else None,
                training_seconds=train['seconds'],warmup_seconds=warmup['seconds'] if warmup else 0.,setup_seconds=setup_seconds,worker_seconds=time.perf_counter()-start,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(a.output/'report.json',report);write(a.output/'worker_ready.json',dict(passed=True,report_sha256=sha(a.output/'report.json')))
    deadline=time.monotonic()+120
    while not (a.output/'release_worker.json').exists():require(time.monotonic()<deadline,'Monitor shutdown handshake timed out');time.sleep(.25)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['gids','digit_full'],required=True);p.add_argument('--model',choices=['sage','gcn','gat'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);a=p.parse_args()
    require(os.geteuid()==0 and __debug__,'Native entry requires local sudo, assertions enabled')
    with open('/tmp/digit-pa-sage-libnvm0.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
