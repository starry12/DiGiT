"""Native resource factory called only inside an admitted, monitored worker.

No standalone unguarded execute path is provided. The controller supplies a live
checkpoint function, verified SSD/source binding and authenticated BFS receipt.
"""
import gc,hashlib,json,os,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def source_arrays(path,nodes,smoke=False):
    from digit_paths import json_loads
    source=json_loads(Path(path).read_text())['dataset']
    if source['num_nodes']!=nodes or source['feature_dim']!=1024:
        raise ValueError('IG source geometry mismatch')
    arrays={}
    for key,shape,fmt in [('labels',(nodes,),'raw_fp32_integral_offset0'),
                          ('feature',(nodes,1024),'raw_row_major_fp32_offset0')]:
        item=source[key];p=Path(item['path']);size=int(np.prod(shape))*4
        if not p.is_absolute() or '${' in str(p):
            raise ValueError('Unresolved IG source path: '+str(p))
        if item['format']!=fmt or item['shape']!=list(shape) or item['bytes']!=size or p.stat().st_size!=size:
            raise ValueError('IG source format or size mismatch: '+key)
        with p.open('rb'):
            pass
        if key=='labels' or smoke:
            arrays[key]=np.memmap(p,dtype='<f4',mode='r',shape=shape)
    return arrays['labels'],arrays.get('feature')

def create(config,arm,*,output,canonical,bfs_receipt,check,mark,profile_directory,smoke=False):
    if os.geteuid()!=0 or not callable(check):raise RuntimeError('Admitted root worker and live guard required')
    check() # existing controller ownership, UUID, resource and kernel guards
    from .common import setup,verify,ROOT,DATA,require,cfg,write,sha,state_hash
    execution=verify();setup()
    p=cfg()
    labels,raw=source_arrays(ROOT/'configs/igb/dataset.json',p['nodes'],smoke)
    mark('source_arrays_verified')
    import torch,dgl,IGPerfNative as native
    from .admission import estimate
    from .features import FormalFeatures
    from candidates.ig_demand_pair_v1.affinity import apply_affinity
    from candidates.ig_demand_pair_v1.sampler import configure
    from candidates.ig_demand_pair_v1.adjacency_sampler import MigrationSampler
    from candidates.ig_demand_pair_v1.cache import overlay
    from candidates.ig_demand_pair_v1.model import make_model,optimizer
    from bounded_io import load_csc
    from digit.artifacts import ArtifactBundle
    from ae.common import check_device,host
    from candidates.unified_training_v2.order import roots_digest
    kind='full' if arm=='digit' else 'gids'
    require(Path(native.__file__).resolve()==Path(__file__).parent/'runtime/IGPerfNative.so','Wrong IG unified backend')
    require(native.IG_ASYNC_ABI==1,'Missing async ABI')
    apply_affinity('digit_full' if arm=='digit' else 'gids')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,9),'Expected one sm89 device')
    check_device();budget=estimate();free,total=torch.cuda.mem_get_info()
    require(free>=budget['required_bytes'] and host()>=budget['host_required_bytes'],'IG unified memory admission')
    torch.cuda.set_per_process_memory_fraction(p['torch_allocator_cap_bytes']/total)
    cm=json.loads((DATA/'csc/manifest.json').read_text());names=['original_indptr.npy','original_indices.npy','original_eids.npy']
    require(bfs_receipt['passed'] and bfs_receipt['exact_permutation'] and len(canonical)==p['train_nodes'],'Full BFS required')
    require(bfs_receipt['orientation']=='incoming_CSC_train_induced' and bfs_receipt['source']==[str(DATA/'csc'/n) for n in names[:2]]
            and len(bfs_receipt['source_file_sha256'])==2,'Wrong BFS source graph/orientation')
    require(sha(bfs_receipt['output'])==bfs_receipt['output_sha256'],'BFS file changed')
    from .bfs import exact_permutation
    exact_permutation(canonical,p['train_nodes'])
    # The canonical order must have been made from the exact graph being loaded.
    for path,h in zip(bfs_receipt['source'],bfs_receipt['source_file_sha256']):
        require(sha(path)==h,'BFS graph source changed')
    graph,csc=load_csc(*[DATA/'csc'/name for name in names],p['nodes'],host_cap=70*2**30,
                       expected_sha256=[cm['files'][name]['payload_sha256'] for name in names])
    require(graph.num_edges()==p['edges'],'Wrong IG graph')
    configure(kind);arrays=metadata=None
    if arm=='digit':
        manifest=json.loads((DATA/'full/manifest.json').read_text())
        arrays={k:np.load(DATA/'full'/v['path'],mmap_mode='r') for k,v in manifest['files'].items() if k!='reordered_features'}
        arrays,manifest=overlay(arrays,manifest)
        sampler=MigrationSampler(p['fanouts'],ArtifactBundle(DATA/'full',manifest,arrays),random_seed=0,host_cap=100*2**30)
        metadata=sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
        from .cache import prepare
        Path(profile_directory).mkdir(parents=True,exist_ok=True)
        hot,profile=prepare(graph,sampler,canonical,p['nodes'],np.load(DATA/'gids_cpu_rows.npy',mmap_mode='r'),profile_directory,check,
                            lambda stage,**kw:mark(stage,**kw),config)
    else:sampler=dgl.dataloading.NeighborSampler(p['fanouts'],replace=False)
    features=FormalFeatures(kind,output,mark)
    if arm=='digit':features.profile_directory=profile_directory;features.profile_config=config;features.set_hot(hot,arrays)
    features.populate(None);check()
    gpu=list(features.store.get_gpu_cache_stats())
    require(gpu[6]==0,'Formal GPU cache must be cold after CPU preload')
    audits=[]
    def audit(inputs,blocks,x):
        if not smoke:return
        ids=inputs.cpu().numpy()
        require(np.array_equal(x.cpu().numpy().view('u4'),np.asarray(raw[ids]).view('u4')),'IG native/source feature mismatch')
        ip,ix,_=graph.adj_tensors('csc')
        for block in blocks:
            u,v=block.edges();src=block.srcdata[dgl.NID][u].cpu().numpy();dst=block.dstdata[dgl.NID][v].cpu().numpy()
            for node in np.unique(dst):
                neighbors=ix[int(ip[node]):int(ip[node+1])].numpy()
                require(np.isin(src[dst==node],neighbors).all(),'Sampled edge outside original CSC')
        audits.append(dict(rows=len(ids),source_features_equal=True,original_edges_verified=True))
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);dgl.seed(0)
    model=make_model('sage','cuda');opt=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['data']['weight_decay'])
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);dgl.seed(0)
    if arm=='digit':sampler.random_seed=0;sampler._cuda_call_counter=0
    canonical_hash=roots_digest(canonical[i:i+1048576] for i in range(0,len(canonical),1048576))
    receipt=dict(source_sha256=execution,graph_sha256=hashlib.sha256(json.dumps(cm,sort_keys=True).encode()).hexdigest(),
                 split_sha256=hashlib.sha256(('IG:train:[0,%d)'%p['train_nodes']).encode()).hexdigest(),
                 canonical_sha256=canonical_hash,cpu_feature_bytes=p['cpu_cache_rows']*4096,gpu_feature_bytes=4*2**30,
                 worker_id=str(os.getpid()),evidence_scope='native',roots_verified_exact=True,gpu_resident_rows_start=gpu[6],
                 native_acceptance_required=True)
    def close():
        torch.cuda.synchronize()
        if arm=='digit':sampler.close()
        # Feature-store and registered graph owners remain live until worker exit.
        # Parent controller must verify normal exit and GPU release before acceptance.
        check()
    def window_evidence(epoch,phase,raw):
        with (Path(output)/'io_windows.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,phase=phase,**raw))+'\n')
    return SimpleNamespace(model=model,optimizer=opt,graph=graph,sampler=sampler,features=features,labels=labels,
                           output=Path(output),check=check,close=close,window_evidence=window_evidence,receipt=receipt,
                           owners=(arrays,metadata,csc),audit=audit,audits=audits)
