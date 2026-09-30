"""Adapt accepted preparation to v6 without relabeling prior workers or failures."""
import copy
from pathlib import Path
from .common import HERE,read,write,sha,identity,require
from .protocol import validate


def review(p,protocol_path,origin=None):
    validate(p)
    origin=read(HERE/'preparation_origin.json') if origin is None else origin
    from candidates.pa_sage_cache_policy_v3.common import verify as verify_old
    from candidates.pa_sage_cache_policy_v3.binding import check as old_binding
    from candidates.pa_sage_cache_policy_v3.prepared import check as old_prepared
    from candidates.pa_sage_cache_policy_v2.common import binary_receipt
    require(verify_old()==origin['source_sha256'],'Prior controller identity changed')
    for desc in (origin['status'],origin['protocol']):
        require(sha(desc['path'])==desc['sha256'],'Prior run/protocol changed')
    state=read(origin['status']['path']);previous_protocol=Path(origin['protocol']['path'])
    prior=read(previous_protocol)
    require(state['source_sha256']==origin['source_sha256'] and
            state['protocol_sha256']==sha(previous_protocol) and
            state['completed']==origin['completed'],'Preparation origin changed')
    require(set(state['completed'])=={'bind','build','rank','profile','prepare'},
            'Reuse only complete preparation, never failed smoke or measured training')
    comparable=copy.deepcopy(p)
    for key in ('policy_decision','capacity_note'):
        comparable[key]=prior[key]
    for arm in comparable['arms']:
        for key in ('gpu_policy','gpu_feature_cache_bytes','lookup_order'):
            comparable['arms'][arm][key]=prior['arms'][arm][key]
    require({k:v for k,v in prior.items() if k not in ('execution','implementation')}==
            {k:v for k,v in comparable.items() if k not in ('execution','implementation')},'Preparation workload differs')
    for desc in origin['completed'].values():
        require(sha(desc['path'])==desc['sha256'] and read(desc['path'])['passed'],'Prior accepted stage changed')
    binding_path=Path(origin['completed']['bind']['path']);bound=read(binding_path)
    old_binding(bound,previous_protocol,origin['source_sha256'])
    prepared_path=Path(origin['completed']['prepare']['path']);prepared=read(prepared_path)
    old_prepared(prepared,prior,previous_protocol,binding_path,origin['source_sha256'])
    require(read(origin['completed']['build']['path'])['build_receipt']==binary_receipt(),'Compiled backend changed')
    profile=read(origin['completed']['profile']['path'])
    require(profile['worker_returncode']==0 and profile['monitor']['passed'] and
            profile['batches']==p['profile']['batches'] and profile['seed']==p['profile']['seed'] and
            profile['feature_reads']==profile['optimizer_updates']==profile['evaluation_calls']==0,
            'Independent preprofile was not accepted')
    for path,desc in {item['path']:item for item in [prepared['counts'],*prepared['arms'].values()]}.items():
        require(sha(path)==desc['sha256'] and identity(path)==desc['identity'],'Prepared counts/hot nodes changed')
    return origin,bound,prepared


def materialize(p,protocol_path,attempt,execution):
    origin,bound,prepared=review(p,protocol_path)
    # These are adaptation receipts. Original accepted source/protocol hashes stay
    # in inherited_from and the original files; no old worker becomes a v6 run.
    bound=copy.deepcopy(bound)
    bound.update(schema='digit-cache-input-binding-v6',source_sha256=execution,
        protocol_sha256=sha(protocol_path),inherited_from=origin['completed']['bind'])
    bound['files'][str(protocol_path.resolve())]=dict(identity=identity(protocol_path),
        sha256=sha(protocol_path),full_hash_checked=True)
    binding_path=attempt/'inputs.json';write(binding_path,bound)
    prepared=copy.deepcopy(prepared)
    prepared.update(schema='digit-cache-prepared-v6',source_sha256=execution,
        protocol_sha256=sha(protocol_path),binding_sha256=sha(binding_path),
        inherited_from=origin['completed']['prepare'],original_profile_monitoring='accepted nvidia-smi v3 profile')
    prepared_path=attempt/'prepared.json';write(prepared_path,prepared)
    result=dict(passed=True,kind='adapted_accepted_preparation',source_sha256=execution,
        protocol_sha256=sha(protocol_path),origin_sha256=sha(HERE/'preparation_origin.json'),
        binding=dict(path=str(binding_path),sha256=sha(binding_path)),
        prepared=dict(path=str(prepared_path),sha256=sha(prepared_path)),
        reused=['bind','rank','profile','prepare'],new_backend_build_required=True,smoke_or_full_results_reused=False)
    path=attempt/'reuse.json';write(path,result);return path


def check(path,p,protocol_path,execution):
    from .binding import check as check_binding
    from .prepared import check as check_prepared
    value=read(path)
    require(value['passed'] and value['source_sha256']==execution and
            value['protocol_sha256']==sha(protocol_path) and
            value['origin_sha256']==sha(HERE/'preparation_origin.json') and
            value['smoke_or_full_results_reused'] is False,'Adaptation receipt changed')
    for key in ('binding','prepared'):
        require(sha(value[key]['path'])==value[key]['sha256'],'Adapted input changed')
    check_binding(read(value['binding']['path']),protocol_path,execution)
    check_prepared(read(value['prepared']['path']),p,protocol_path,value['binding']['path'],execution)
    return Path(value['binding']['path']),Path(value['prepared']['path'])
