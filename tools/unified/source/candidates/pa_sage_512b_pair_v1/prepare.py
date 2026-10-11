"""Reuse verified directed metadata and SSD data; select the exact CPU budget."""
from .common import *


def main():
    import numpy as np
    from candidates.pa_sage_cache_policy_v1.selection import topk
    require(not (OUT/'inputs.json').exists(),'Preserve input binding')
    require(not LARGE.exists(),'Preserve large-file output')
    parents()
    baseline=read(legacy.OUT/'inputs.json');check_inputs(baseline)
    old_path=ROOT/'results/pa_sage_ablation_cache_20260924_v3/resume2/native/inputs.json'
    old=read(old_path);base=ROOT/'data/papers_g2_random_v2';bundle=base/'final/bundle'
    manifest=read(bundle/'manifest.json');p=read(HERE/'protocol.json')
    require(manifest['dataset']['num_edges']==p['edges'] and manifest['dataset']['num_nodes']==p['nodes'],'Wrong graph layout')
    for name in ('indptr','indices'):
        require(manifest['source'][name]['sha256']==baseline['files'][baseline['csc'][name]]['sha256'],'Different source graph')
    require(manifest['source']['features']['sha256']==baseline['files'][baseline['features']]['sha256'],'Different source features')
    files=dict(baseline['files'])
    def bind(path, expected=None, prior=None):
        path=Path(path);before=identity(path)
        if prior:
            require(before==prior['identity'],'Inherited file changed: '+str(path));digest=prior['sha256']
        else:
            progress(OUT,'binding',path=str(path));digest=sha(path)
        require(expected is None or digest==expected,'Checksum mismatch: '+str(path))
        require(identity(path)==before,'Input changed while binding')
        files[str(path)]=dict(identity=before,sha256=digest,inherited=bool(prior))
    for name,desc in manifest['files'].items():
        path=bundle/desc['path'];key=str(path.relative_to(ROOT))
        bind(path,desc['sha256'],old['files'].get(key))
    bind(bundle/'manifest.json',prior=old['files'][str((bundle/'manifest.json').relative_to(ROOT))])
    for rel in ('prepared.json','final/descriptor.json','final/validation.json','hot_ready.json','ssd_ready.json'):
        bind(base/rel)
    prepared=read(base/'prepared.json');validation=read(base/'final/validation.json')
    require(validation['passed'] and validation['expanded_adjacency_multiset_exact'] and validation['storage_inverse_exact'],'Unverified directed bundle')
    for rel in ('final/descriptor.json','final/validation.json','hot_ready.json','final/bundle/manifest.json'):
        require(sha(base/rel)==prepared['bindings'][rel],'Prepared receipt mismatch')
    payload=read(base/'ssd_ready.json')
    require(payload['full_readback_passed'] and payload['passed'] and payload['manifest_sha256']==sha(bundle/'manifest.json'),'Unverified grouped payload')
    for path,digest in [(payload['state'],payload['state_sha256']),(payload['verify_receipt'],payload['verify_receipt_sha256'])]:bind(path,digest)
    profile=read(base/'hot_ready.json');freqpath=base/'logical_frequency.npy'
    bind(freqpath,profile['frequency_sha256'])
    freq=np.load(freqpath,mmap_mode='r');require(freq.shape==(p['nodes'],),'Wrong frequency extent')
    hot=topk(freq,p['cpu_rows'])
    oldhot=np.load(base/'hot_nodes.npy',mmap_mode='r')
    require(len(hot)==p['cpu_rows'],'Wrong exact CPU budget')
    require(np.isin(oldhot,hot,assume_unique=True).all(),'New budget must retain the previously ranked directed hot set')
    LARGE.mkdir(parents=True)
    hotpath=LARGE/'digit_hot_nodes.npy';np.save(hotpath,hot);bind(hotpath)
    result=dict(baseline,files=files,bundle=str(bundle),bundle_manifest_sha256=sha(bundle/'manifest.json'),
        digit_payload=payload,digit_hot_path=str(hotpath),digit_hot_sha256=sha(hotpath),
        digit_profile=dict(receipt=str(base/'hot_ready.json'),batches=profile['batches'],
            previous_hot_rows=len(oldhot),new_hot_rows=len(hot),additional_rows=len(hot)-len(oldhot),
            selection='descending frequency, ascending logical ID at ties; reuse directed profile, no measured-run data'),
        inherited_bundle_binding=dict(path=str(old_path),sha256=sha(old_path)),prepared_unix=time.time())
    check_inputs(result);write(OUT/'inputs.json',result)
    progress(OUT,'inputs_prepared',native_started=False,cpu_rows=len(hot),raw_ssd_writes=False)


if __name__=='__main__':main()
