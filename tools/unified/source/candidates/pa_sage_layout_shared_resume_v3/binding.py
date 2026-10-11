"""Bind verified metadata and read-only pool evidence, with no per-point payload."""
from .common import *
from .settings import check_manifest
from .pool import spec,check_receipt,identity

def input_binding(output,execution):
    p=cfg();base=Path(p['base_layout']);overlay=Path(p['overlay']);data=ROOT/p['data'];m=read(base/'final/bundle/manifest.json')
    from .graph_binding import descriptors
    descriptors(data,p['graph'])  # Reject schema/header mistakes before expensive file hashing.
    check_manifest(p,m);build=read(base/'build_receipt.json');over=read(overlay/'overlay_receipt.json')
    require(build['passed'] and over['passed'] and not build['fixture'] and not over['fixture'],'Invalid/fixture preparation')
    require(build['point']==over['point']==p['point'],'Wrong layout point')
    require(build['manifest_sha256']==over['layout_manifest_sha256']==sha(base/'final/bundle/manifest.json'),'Layout identity changed')
    require(over['nodes']==p['graph']['nodes'] and over['edges']==p['graph']['edges'],'Wrong logical graph')
    contract=read(base/'contract.json')
    require(build['input_binding_sha256']==sha(base/'contract.json') and contract['point']==p['point'],'Layout contract changed')
    require(contract['inputs']['hot_nodes']['sha256']==p['hot_sha256'] and contract['hot_rows']==p['cpu_cache_rows'],'Logical hot set changed')
    pool=spec(p['feature_mode']);receipt=check_receipt(p['feature_mode'])
    require(m['feature']['num_storage_rows']*512<=receipt['verified_bytes'],'Unverified/out-of-range layout addresses')
    if p['feature_mode']=='real':
        require(m['files']['reordered_features']['sha256']==pool['source_full_sha256'],'Real layout differs from resident feature payload')
    files={}
    def add(path,expected=None,full_hash=None):
        path=Path(path).resolve();before=identity(path)
        write(output/'binding_progress.json',dict(path=str(path),bytes=before['bytes'],completed_files=len(files),updated_unix=time.time()))
        if full_hash is None:full_hash=p['point']['id']=='g2_r20' or before['bytes']<=1024**2
        digest=sha(path) if full_hash else None
        require(identity(path)==before and (not full_hash or expected is None or expected==digest),'Input hash/identity changed: '+str(path))
        files[str(path)]=dict(sha256=digest,identity=before,expected_sha256=expected,content_hash_checked=full_hash)
    add(P);add(HERE/'pools.json');add(pool['state'],pool['state_sha256']);add(receipt['receipt_path'])
    # Source bytes were compared with live SSD bytes and hashed once in pool
    # verification. Keep that proof plus file identity, not a per-worker TB hash.
    add(pool['source'],full_hash=False)
    for rel in ('contract.json','build_receipt.json','final/bundle/manifest.json','final/descriptor.json','final/validation.json','full_cpu_rows.npy'):
        add(base/rel,build['full_cpu_rows_sha256'] if rel=='full_cpu_rows.npy' else None)
    for name,entry in m['files'].items():
        if name!='reordered_features':add(base/'final/bundle'/entry['path'],entry['sha256'])
    add(base/'hot_nodes.npy',p['hot_sha256']);add(p['hot_nodes'],p['hot_sha256']);add(p['orders_file'])
    add(overlay/'overlay_receipt.json')
    for name,digest in over['files'].items():add(overlay/name,digest)
    add(data/'prepared.json')
    from .graph_binding import bind
    csc_binding=bind(data,p['graph'],add)
    source=source_config()
    for desc in (source['label_identity'],source['source_contract']['original_edges'],source['splits']['train']):add(desc['path'],desc['sha256'])
    if p['feature_mode']=='real':add(source['source_features']['path'],source['source_features']['sha256'])
    value=dict(candidate_sha256=execution,protocol_sha256=sha(P),prepared_sha256=sha(overlay/'overlay_receipt.json'),
          csc_binding=csc_binding,verification_policy=p['verification_policy'],point=p['point'],layout_manifest_sha256=sha(base/'final/bundle/manifest.json'),files=files,
          feature_mode=p['feature_mode'],pool_receipt_sha256=sha(receipt['receipt_path']))
    write(output/'inputs.json',value);return value
