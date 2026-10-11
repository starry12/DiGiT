"""Reuse verified IG data, hash metadata; keep TB payload verification scope explicit."""
import math
import numpy as np
from candidates.ig_perf_v5.common import *
from ae.common import check_payload

def payload_sha(path,offset=0):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        f.seek(offset)
        for blob in iter(lambda:f.read(8*2**20),b''):h.update(blob)
    return h.hexdigest()

def prepare_orders():
    started=time.perf_counter();p=cfg();folder=ROOT/p['train_order'];ready=folder.parent/'ready.json'
    if ready.exists():
        v=read(ready);require(v['protocol_sha256']==sha(P),'Order protocol changed')
        for name,digest_ in v['files'].items():require(sha(folder.parent/name)==digest_,'Prepared order changed')
        require(identity(ROOT/p['source_train_order'])==v['source_identity'],'Source order identity changed')
        return v
    source_ready=read(ROOT/p['source_order_ready'])
    require(source_ready['passed'] and source_ready['protocol_sha256']==p['source_order_protocol_sha256'],'Unverified source order')
    source=ROOT/p['source_train_order'];before=identity(source)
    require(sha(source)==source_ready['files'][source.name],'Source order changed')
    order=np.load(source,mmap_mode='r');require(order.shape==(p['train_nodes'],) and order.dtype==np.dtype('<i8'),'Wrong source order')
    roots=np.array(order[:p['total_train_batches']*p['batch_size']],dtype='<i8');require(identity(source)==before,'Source changed')
    require(len(np.unique(roots))==len(roots) and roots.min()>=0 and roots.max()<p['train_nodes'],'Invalid short root order')
    folder.parent.mkdir(parents=True,exist_ok=True)
    with folder.open('xb') as f:np.save(f,roots)
    split=p['warmup_batches']*p['batch_size']
    v=dict(passed=True,preparation_seconds=time.perf_counter()-started,protocol_sha256=sha(P),source_order=str(source),source_identity=before,source_file_sha256=source_ready['files'][source.name],train_payload_sha256=digest(roots),warmup_payload_sha256=digest(roots[:split]),training_payload_sha256=digest(roots[split:]),smoke_payload_sha256=digest(roots[:p['smoke_train_batches']*p['batch_size']]),training_window_payload_sha256=[digest(roots[split+j*p['window_batches']*p['batch_size']:split+(j+1)*p['window_batches']*p['batch_size']]) for j in range(p['measurement_windows'])],files={folder.name:sha(folder)})
    write(ready,v);return v

def bind(output,execution,preflight=False):
    p=cfg();output=Path(output);files={}
    def add(path,expected=None,offset=0,hash_now=True):
        path=Path(path);before=identity(path);digest_=payload_sha(path,offset) if hash_now else expected
        require(identity(path)==before,'Input changed while reading')
        if expected is not None and hash_now:require(digest_==expected,'Input SHA differs: '+str(path))
        try:key=str(path.relative_to(ROOT))
        except ValueError:key=str(path)
        files[key]=dict(identity=before,sha256=digest_,hash_offset=offset,content_hash_checked_now=hash_now)
    if not preflight:
        pre=ROOT/'results/ig_perf_preflight_20260922_v5/check';s=read(pre/'status.json');require(s['passed'] and s['complete'] and s['candidate_sha256']==execution,'IG preflight incomplete/stale')
        require(s['entry_manifest_sha256']==sha(ROOT/'submission/v9/manifest.json'),'Entry preflight differs')
        for name,digest_ in s['evidence_sha256'].items():require(sha(pre/name)==digest_,'Preflight evidence changed')
        value=read(pre/'inputs.json');check_inputs(value);files.update(value['files'])
        for name in ('status.json','inputs.json','model_checks.json','cuda_checks.json','memory_checks.json','admission.json','environment.json','monitor_checks.json','monitor_review_checks.json','candidates_ig_monitor_v1_tests.log','candidates_ig_monitor_v1_review_tests.log'):add(pre/name)
        value=dict(value,files=files,preflight_sha256=sha(pre/'status.json'));write(output/'inputs.json',value);return value
    prepared=prepare_orders()
    add(ROOT/p['train_order'])
    add((ROOT/p['train_order']).parent/'ready.json')
    for name in ('dataset.json','gids_cache.json','full_cache.json','ssd_plan.json'):add(ROOT/'configs/igb'/name)
    for name in ('full/manifest.json','full/metadata_ready.json','csc/manifest.json'):add(DATA/name)
    m=read(DATA/'full/manifest.json');require(m['grouping']['group_size']==p['group_size'] and m['feature']['row_bytes']==4096,'Wrong IG geometry')
    require(m['dataset']['num_nodes']==p['nodes'] and m['dataset']['num_edges']==p['edges'],'Wrong IG graph')
    # Manifest SHA covers each whole NPY file, including header.
    # Metadata is rehashed now; 1.3TB reordered features retain the existing full
    # readback hash plus current file identity, and native smoke compares real rows.
    for name,desc in m['files'].items():
        path=DATA/'full'/desc['path'];a=np.load(path,mmap_mode='r');require(list(a.shape)==desc['shape'] and a.dtype.str==desc['dtype'],'Wrong IG array header')
        require(path.stat().st_size==a.offset+a.nbytes,'Wrong array size');off=a.offset;del a
        progress(output,'checking_metadata',file=name)
        add(path,desc['sha256'],0,hash_now=name!='reordered_features')
    cm=read(DATA/'csc/manifest.json');require(cm['accepted'] and cm['num_nodes']==p['nodes'] and cm['num_edges']==p['edges'] and cm['all_endpoints_checked'] and cm['eid_permutation_verified'],'Unverified CSC provenance')
    for name,desc in cm['files'].items():
        progress(output,'checking_csc',file=name);add(DATA/'csc'/name,desc['file_sha256'])
    source=read(ROOT/'configs/igb/dataset.json')['dataset']
    for key in ('feature','labels'):
        desc=source[key];require(identity(desc['path'])['bytes']==desc['bytes'],'Wrong raw source size');add(desc['path'],desc['sha256'],hash_now=key=='labels')
    cache=read(ROOT/'configs/igb/gids_cache.json');require(Path(cache['path']).stat().st_size==p['cpu_cache_rows']*4096,'Wrong CPU cache size')
    add(cache['path'],cache['sha256'],hash_now=False);add(DATA/'gids_cpu_rows.npy',cache['rows_binding']['sha256'])
    # Cache bytes are fully hashed again while filling each native worker.
    plans=read(ROOT/'configs/igb/ssd_plan.json')
    payloads={}
    for arm in ('gids','full'):
        v=check_payload('igb_'+arm);plan=plans[arm]
        require(v['offset']==plan['offset'] and v['bytes']==plan['bytes'] and v['file_sha256']==plan['file_sha256'],'SSD plan differs')
        add(v['state'],v['state_sha256']);payloads[arm]=v
    require(plans['gids']['end']<=plans['full']['offset'],'IG payloads overlap')
    add(ROOT/'data/papers_g2_random_v2/ssd_ready.json');pa=read(ROOT/'data/papers_g2_random_v2/ssd_ready.json');require(plans['full']['end']<=pa['offset'],'IG overlaps accepted PA g2')
    value=dict(candidate_sha256=execution,protocol_sha256=sha(P),files=files,prepared_orders=prepared,ssd=payloads,
               large_payloads_freshly_hashed=False,metadata_and_csc_freshly_hashed=True,
               note='TB feature payload hashes and SSD full-readback receipts are inherited, checked against frozen manifests/current identities; native smoke source comparisons still required. CPU cache payload fully hashed during each worker preload.')
    write(output/'inputs.json',value);return value
