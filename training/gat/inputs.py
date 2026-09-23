"""Validate and reuse the frozen PA data; never re-sign its SAGE preparation receipt."""
from training.gat.common import *
from training.sage.worker import identity,check_inputs

def compatibility():
    p=cfg();parent=read(PARENT/'protocol.json');base=ROOT/p['base_layout'];data=ROOT/p['data']
    keys=('graph','fanouts','batch_size','gpu_cache_bytes','cpu_cache_rows','base_layout','data','validation_trace','test_trace','metadata_mode','bfs_enabled','layout_policy')
    for key in keys:require(p[key]==parent[key],'Data reuse not validated: '+key)
    require(p['data_preparation_protocol_sha256']==sha(PARENT/'protocol.json'),'Wrong data preparation protocol')
    ready=read(data/'prepared.json');manifest=read(base/'final/bundle/manifest.json')
    require(ready['passed'] and ready['protocol_sha256']==p['data_preparation_protocol_sha256'],'Parent preparation changed')
    require(ready['base_manifest_sha256']==sha(base/'final/bundle/manifest.json'),'Parent layout changed')
    require(manifest['grouping']['group_size']==2 and p['fanouts'][0]%2==0,'Incompatible group size/fanout')
    require(manifest['feature']['dim']==p['input_dim']==128,'Wrong feature dimension')
    require(len(p['fanouts'])==p['layers']==3 and p['epochs']==20 and p['seed']==0,'Unsupported protocol extent')
    require(p['model_config']==read(ROOT/'configs/papers/contract.json')['models']['gat'],'Wrong historical GAT contract')
    for key,val in read(ROOT/'configs/papers/contract.json')['optimizer'].items():require(p['optimizer']['kwargs'][key]==val,'Wrong historical optimizer: '+key)
    return dict(passed=True,group_size=2,fanouts=p['fanouts'],same_data_graph_and_traces=True,new_ssd_writes=False,
                parent_data_protocol_sha256=p['data_preparation_protocol_sha256'])

def input_binding(output,execution,require_preflight=True):
    from training.sage.run import input_binding as parent_binding
    compatibility();output=Path(output)
    if require_preflight:
        pre=Path(os.environ.get('DIGIT_ARTIFACT_PREFLIGHT',str(ROOT/cfg()['preflight_output'])));status=read(pre/'status.json')
        require(status['passed'] and status['complete'] and status['candidate_sha256']==execution,'GAT preflight incomplete or stale')
        require(status['entry_manifest_sha256']==sha(ROOT/'evaluation/gat/manifest.json'),'Preflight entry version differs')
        require(status['model_checks']['passed'] and status['cuda_checks']['passed'] and status['memory_checks']['passed'] and status['live_admission']['passed'],'GAT model/resource checks failed')
        for name,digest in status['evidence_sha256'].items():require(sha(pre/name)==digest,'Preflight evidence changed')
        check_inputs(read(pre/'inputs.json'))
    value=parent_binding(output,sha(PARENT/'manifest.json'))
    (output/'inputs.json').rename(output/'parent_data_inputs.json')
    parent_sha=sha(output/'parent_data_inputs.json')
    value.update(candidate_sha256=execution,protocol_sha256=sha(P),parent_data_binding_sha256=parent_sha,
                 parent_data_candidate_sha256=sha(PARENT/'manifest.json'),model='gat',compatibility=compatibility())
    if require_preflight:
        for name in ('status.json','inputs.json','model_checks.json','cuda_checks.json','memory_checks.json','admission.json'):
            path=pre/name;value['files'][str(path.relative_to(ROOT))]=dict(sha256=sha(path),identity=identity(path))
    write(output/'inputs.json',value);return value
