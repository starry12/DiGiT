"""Bind each layout, overlay and SSD receipt before spawning a native worker."""
from .common import *
from .settings import check_manifest

def identity(path):
    s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def check_residency(ready):
    registry=ROOT/'ssd_state/libnvm0.offset4398046511104.json'
    require(ready.get('active_registry')==str(registry),'Missing current scratch-slot registry')
    require(ready['offset']==4*2**40 and ready['bytes']<=2**40,'Outside layout scratch slot')
    require(sha(registry)==ready['active_registry_sha256']==ready['state_sha256'],
            'Layout is no longer resident in the SSD scratch slot; provision and verify it again')

def receipt_check(p,fixture=False):
    base=Path(p['base_layout']);overlay=Path(p['overlay']);bundle=base/'final/bundle'
    m=read(bundle/'manifest.json');check_manifest(p,m)
    build=read(base/'build_receipt.json');over=read(overlay/'overlay_receipt.json')
    require(build['passed'] and over['passed'] and build['point']==over['point']==p['point'],'Wrong point receipts')
    require(fixture or (build['fixture'] is False and over['fixture'] is False),'Synthetic layout cannot enter native training')
    require(build['manifest_sha256']==over['layout_manifest_sha256']==sha(bundle/'manifest.json'),'Layout identity mismatch')
    require(over['nodes']==p['graph']['nodes'] and over['edges']==p['graph']['edges'],'Overlay graph differs')
    require(build['input_binding_sha256']==sha(base/'contract.json'),'Layout source contract changed')
    contract=read(base/'contract.json')
    require(contract['point']==p['point'] and contract['hot_rows']==p['cpu_cache_rows'],'Layout contract changed')
    require(contract['inputs']['hot_nodes']['sha256']==p['hot_sha256'],'Different logical CPU hot set')
    require(build['full_cpu_rows_sha256']==sha(base/'full_cpu_rows.npy'),'CPU physical rows changed')
    ready=read(p['ssd_ready'])
    require(ready['passed'] and ready['full_readback_passed'] and ready['device']=='/dev/libnvm0','Missing full SSD readback')
    require(ready['manifest_sha256']==sha(bundle/'manifest.json') and ready['file_sha256']==m['files']['reordered_features']['sha256'],'SSD belongs to another layout')
    require(type(ready['offset']) is int and ready['offset']>=0 and ready['offset']%4096==0,'Wrong SSD alignment')
    require(ready['bytes']==m['feature']['num_storage_rows']*512,'SSD payload size differs')
    require(sha(ready['state'])==ready['state_sha256'] and sha(ready['verify_receipt'])==ready['verify_receipt_sha256'],'SSD evidence changed')
    for name in ('state','verify_receipt'):
        r=read(ready[name])
        require(r['status']=='verified' and r['device']==ready['device'] and r['device_offset_bytes']==ready['offset'] and
                r['payload_bytes']==ready['bytes'] and r['feature_file_sha256']==ready['file_sha256'],'SSD state/receipt mismatch')
    if not fixture:
        check_residency(ready)
    return m,build,over,ready

def input_binding(output,execution):
    p=cfg();m,build,over,ready=receipt_check(p);base=Path(p['base_layout']);overlay=Path(p['overlay']);data=ROOT/p['data'];files={}
    def add(path,expected=None):
        path=Path(path).resolve();before=identity(path)
        write(output/'binding_progress.json',dict(path=str(path),bytes=before['bytes'],completed_files=len(files),updated_unix=time.time()))
        digest=sha(path);require(identity(path)==before and (expected is None or expected==digest),'Input changed/mismatch: '+str(path))
        files[str(path)]=dict(sha256=digest,identity=before)
    graph=read(data/'prepared.json');require(graph['passed'],'Graph not prepared')
    for name,digest in graph['bindings'].items():
        # Only the shared logical graph is reused; g2 reordered arrays are not inputs to this point.
        if name.startswith('original_'):add(data/name,digest)
    for name,digest in p['graph']['csc_sha256'].items():
        require(files[str((data/('original_'+name+'.npy')).resolve())]['sha256']==digest,'Wrong shared logical graph')
    add(data/'prepared.json');add(P)
    for rel in ('contract.json','build_receipt.json','final/bundle/manifest.json','full_cpu_rows.npy','ssd_ready.json'):add(base/rel)
    for entry in m['files'].values():add(base/'final/bundle'/entry['path'],entry['sha256'])
    add(base/'hot_nodes.npy',p['hot_sha256']);add(p['hot_nodes'],p['hot_sha256']);add(p['orders_file'])
    add(overlay/'overlay_receipt.json')
    for name,digest in over['files'].items():add(overlay/name,digest)
    for name in ('state','verify_receipt'):add(ready[name],ready[name+'_sha256'])
    add(ready['active_registry'],ready['active_registry_sha256'])
    source=source_config()
    for desc in (source['source_features'],source['label_identity'],source['source_contract']['original_edges']):add(desc['path'],desc['sha256'])
    desc=source['splits']['train'];add(desc['path'],desc['sha256'])
    value=dict(candidate_sha256=execution,protocol_sha256=sha(P),prepared_sha256=sha(overlay/'overlay_receipt.json'),
               point=p['point'],layout_manifest_sha256=sha(base/'final/bundle/manifest.json'),files=files)
    write(output/'inputs.json',value);return value
