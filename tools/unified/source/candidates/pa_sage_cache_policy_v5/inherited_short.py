"""Keep v4 native short reports under their original identities, not relabeled v5."""
from pathlib import Path
from .common import ROOT,HERE,read,sha,write,require


def verify_change_scope():
    previous=(ROOT/'candidates/pa_sage_cache_policy_v3/training.py').read_text()
    expected=previous.replace('from candidates.pa_sage_cache_policy_v2.counters import snapshot,interval',
        'from candidates.pa_sage_cache_policy_v5.counters import snapshot,interval')
    expected=expected.replace('from .trace import Trace,add_batch','from candidates.pa_sage_cache_policy_v3.trace import Trace,add_batch')
    expected=expected.replace('from .native_support import model,train_ids','from candidates.pa_sage_cache_policy_v3.native_support import model,train_ids')
    require((HERE/'training.py').read_text()==expected,'Training changed beyond the counter import')
    old=(ROOT/'candidates/pa_sage_cache_policy_v2/counters.py').read_text()
    expected=old.replace('from .common import require, SCRATCH_BYTES','from candidates.pa_sage_cache_policy_v2.common import require, SCRATCH_BYTES')
    expected=expected.replace("device_io_interval(before['device'], after['device'])", "device_io_interval(before['device'], after['device'], complete_region=complete_region)")
    require((HERE/'counters.py').read_text()==expected,'Native counter collection changed')
    old=(ROOT/'candidates/pa_sage_cache_policy_v4/worker.py').read_text()
    expected=old.replace('from candidates.pa_sage_cache_policy_v3.training import run','from .training import run')
    require((HERE/'worker.py').read_text()==expected,'Worker changed beyond the training import')


def validate_origin(p):
    from candidates.pa_sage_cache_policy_v4.common import verify as verify_prior
    from candidates.pa_sage_cache_policy_v4.validation import smoke_matrix as prior_check
    origin=read(HERE/'short_origin.json');verify_change_scope()
    require(verify_prior()==origin['source_sha256'],'Accepted short controller changed')
    for key in ('status','protocol','short_matrix','prepared'):
        require(sha(origin[key]['path'])==origin[key]['sha256'],'Original short evidence changed: '+key)
    prior=read(origin['protocol']['path']);value=read(origin['short_matrix']['path'])
    require({k:v for k,v in p.items() if k not in ('execution','implementation')}==
            {k:v for k,v in prior.items() if k not in ('execution','implementation')},'Native short workload differs')
    require(p['execution']['monitor']==prior['execution']['monitor']=='disabled' and
            p['execution']['smoke_updates']==prior['execution']['smoke_updates']==4,'Short verification policy differs')
    prior_check(value,prior,origin['source_sha256'],origin['protocol']['sha256'],origin['prepared']['sha256'])
    state=read(origin['status']['path'])
    for arm,desc in value['reports'].items():
        require(state['completed']['smoke_'+arm]==desc,'Short matrix is not accepted in the original journal')
    from candidates.pa_sage_cache_policy_v2.common import binary_receipt
    require(value['backend_binary_sha256']==binary_receipt()['binary_sha256'],'Short backend changed')
    return origin,value


def make(path,p,execution,protocol_sha,prepared_sha):
    origin,prior=validate_origin(p)
    value=dict(schema='digit-cache-inherited-short-matrix-v5',passed=True,source_sha256=execution,
        protocol_sha256=protocol_sha,prepared_sha256=prepared_sha,
        inherited_from=origin['short_matrix'],origin_sha256=sha(HERE/'short_origin.json'),
        backend_binary_sha256=prior['backend_binary_sha256'],reports=prior['reports'],
        fresh_native_smoke=False,scope='v4 four-arm native correctness accepted; v5 changes host counter window arithmetic only')
    path=Path(path)
    if path.exists():require(read(path)==value,'Inherited short receipt changed')
    else:write(path,value)
    return path


def check(value,p,execution,protocol_sha,prepared_sha):
    origin,prior=validate_origin(p)
    require(value['schema']=='digit-cache-inherited-short-matrix-v5' and value['passed'] and
            value['source_sha256']==execution and value['protocol_sha256']==protocol_sha and
            value['prepared_sha256']==prepared_sha and value['origin_sha256']==sha(HERE/'short_origin.json'),
            'Inherited short receipt context changed')
    require(value['inherited_from']==origin['short_matrix'] and value['reports']==prior['reports'] and
            value['backend_binary_sha256']==prior['backend_binary_sha256'] and value['fresh_native_smoke'] is False,
            'Do not relabel inherited smoke as a fresh v5 run')
    return value
