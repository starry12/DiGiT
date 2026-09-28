"""Reuse verified IG data, hash metadata; keep TB payload verification scope explicit."""
import math
import numpy as np
from candidates.ig_perf_window300_v1.common import *
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
    """Reuse accepted graph/payload evidence, bind a fresh 320-batch root prefix."""
    from candidates.ig_perf_v5.inputs import bind as parent_bind
    from candidates.ig_perf_window300_v1.windows import root_slices
    require(not preflight,'No new large graph scan in this protocol-only revision')
    root_slices(cfg())
    output=Path(output)
    value=parent_bind(output,PARENT_SHA)
    old=dict(value['prepared_orders'])
    prepared=prepare_orders()
    require(prepared['source_identity']==old['source_identity'] and
            prepared['source_file_sha256']==old['source_file_sha256'],
            'Source training order differs from accepted IG')
    require(prepared['warmup_payload_sha256']==old['warmup_payload_sha256'] and
            prepared['smoke_payload_sha256']==old['smoke_payload_sha256'],
            'Warmup/smoke roots changed')
    new_order=ROOT/cfg()['train_order']
    old_order=ROOT/'data/ig_window_v2/train_order.npy'
    new_roots=np.load(new_order,mmap_mode='r')
    old_roots=np.load(old_order,mmap_mode='r')
    require(np.array_equal(new_roots[:len(old_roots)],old_roots),'Old 110-batch prefix changed')
    for path in (new_order,new_order.parent/'ready.json'):
        before=identity(path);digest_=sha(path)
        require(identity(path)==before,'Order changed during binding')
        value['files'][str(path.relative_to(ROOT))]=dict(identity=before,sha256=digest_,
                                                       hash_offset=0,content_hash_checked_now=True)
    value.update(candidate_sha256=execution,protocol_sha256=sha(P),prepared_orders=prepared,
        parent_candidate_sha256=PARENT_SHA,old_110_batch_prefix_preserved=True,
        metadata_and_csc_freshly_hashed=False,metadata_and_csc_preflight_reused=True,
        note='Accepted v5 graph/native/max-batch checks reused with unchanged file identities; only root count/window length changed. No new GPU or SSD acceptance is claimed by this binding. Native paired smoke remains mandatory.')
    write(output/'inputs.json',value)
    return value
