"""Seal graph rankings and one independent frequency profile for all four workers."""
from pathlib import Path
from .common import ARMS, read, sha, require, write, identity


def seal(p, protocol_path, binding_path, ranking_path, profile_path, output, execution):
    ranks,profile=read(ranking_path),read(profile_path)
    for value in (ranks,profile):
        require(value['passed'] and value.get('fixture') is False and value['source_sha256']==execution and
                value['protocol_sha256']==sha(protocol_path) and value['binding_sha256']==sha(binding_path),
                'Preparation belongs to a different workload/version')
    require(profile['kind']=='independent_native_presampling' and profile['native'] and
            profile['worker_returncode']==0 and profile['monitor']['passed'] and profile['monitor']['backend']=='nvidia-smi',
            'Frequency source was not accepted native presampling')
    require(profile['seed']==p['profile']['seed']!=p['seed'] and profile['batches']==p['profile']['batches'] and
            profile['optimizer_updates']==profile['evaluation_calls']==profile['feature_reads']==0 and not profile['raw_ssd_access'],
            'Frequency profile leaked measured training/evaluation/feature I/O')
    binding=read(binding_path)
    require(ranks['graph_sha256']==profile['graph_sha256']==binding['graph_sha256'] and
            profile['layout_sha256']==binding['layout_sha256'] and profile['fanouts']==p['fanouts'] and
            profile['batch_size']==p['batch_size'],'Profile/rank workload differs')
    counts=profile['counts']
    require(identity(counts['path'])==counts['identity'] and sha(counts['path'])==counts['sha256'], 'Frequency counts changed')
    hot=dict(ranks['hot'],freq=profile['hot'])
    for name,entry in hot.items():
        require(entry['rows']==p['arms'][name]['cpu_rows'] and identity(entry['path'])==entry['identity'] and
                sha(entry['path'])==entry['sha256'],'Unbound hot set: '+name)
    result=dict(schema='digit-cache-prepared-v3',passed=True,fixture=False,source_sha256=execution,
        protocol_sha256=sha(protocol_path),binding_sha256=sha(binding_path),
        ranking=dict(path=str(ranking_path),sha256=sha(ranking_path)),
        profile=dict(path=str(profile_path),sha256=sha(profile_path)),counts=counts,
        arms={arm:hot['freq' if arm=='digit' else arm] for arm in ARMS})
    require(result['arms']['freq']==result['arms']['digit'],'Freq/DiGiT must share the exact same file')
    write(output,result);return result


def check(prepared,p,protocol_path,binding_path,execution):
    require(prepared['schema']=='digit-cache-prepared-v3' and prepared['passed'] and not prepared['fixture'] and
            prepared['source_sha256']==execution and prepared['protocol_sha256']==sha(protocol_path) and
            prepared['binding_sha256']==sha(binding_path),'Prepared hot sets changed context')
    require(set(prepared['arms'])==set(ARMS) and prepared['arms']['freq']==prepared['arms']['digit'],'Invalid four-arm hot sets')
    for desc in (prepared['ranking'],prepared['profile']):
        require(sha(desc['path'])==desc['sha256'],'Preparation receipt changed')
    require(identity(prepared['counts']['path'])==prepared['counts']['identity'],'Frequency source changed')
    for name,item in prepared['arms'].items():
        require(item['rows']==p['arms'][name]['cpu_rows'] and identity(item['path'])==item['identity'],
                'Selected hot file changed: '+name)
    return prepared
