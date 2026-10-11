"""Fresh mixed-read experiment over the frozen optimized DiGiT reference."""
from candidates.pa_sage_mean_fused_v1.common import *
from candidates.pa_sage_mean_fused_v1 import common as reference
from .native_sources import native_sources

HERE=ROOT/'candidates/pa_sage_mixed_group_v1'
OUT=ROOT/'results/pa_sage_mixed_group_20261010_v1'
MODULE='candidates.pa_sage_mixed_group_v1'
BINARY=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
VARIANTS=('single','mixed')


def check_inputs(binding):
    from .identity import verify_input
    registry=read(HERE/'identity_registry.json')['files']
    require(binding==read(reference.OUT/'inputs.json'),'Input binding differs from reference')
    require(set(registry)==set(binding['files']),'Input registration set changed')
    for path,item in binding['files'].items():verify_input(path,item['identity'],registry[path],identity)


def setup():
    reference.setup()
    sys.path.insert(0,str(HERE/'runtime'))


def verify():
    m=read(HERE/'manifest.json')
    require(reference.verify()==m['reference_sha256'],'Optimized reference changed')
    for rel,digest in m['files'].items():require(sha(HERE/rel)==digest,'Candidate changed: '+rel)
    require(native_sources()==m['native_sources'],'Native source changed')
    require(sha(BINARY)==m['binary_sha256'],'Mixed backend changed')
    require(sha(OUT/'inputs.json')==m['inputs_sha256']==sha(reference.OUT/'inputs.json'),'Inputs differ')
    p=read(HERE/'protocol.json');old=read(reference.HERE/'protocol.json')
    keys={'scope','mixed_io','group_read_policy','variants','full_order','poll_seconds'}
    require({k:v for k,v in p.items() if k not in keys}=={k:v for k,v in old.items() if k not in keys},'Unrelated training setting changed')
    return sha(HERE/'manifest.json')


def validate_layout(bundle):
    import numpy as np
    m=bundle.manifest
    require(m['grouping']['group_size']==2,'Expected g2 layout')
    bases=bundle.arrays['group_storage_base']
    for lo in range(0,len(bases),1048576):
        b=bases[lo:lo+1048576]
        require(np.all(b%8==0) and np.all(b>=0) and np.all(b+1<m['feature']['num_storage_rows']),
                'Expected two adjacent members in each 4KiB-aligned PA slot')


def check_pair_counts(counts,device,mixed,require_pair=True):
    require(len(counts)==9 and all(type(v) is int and v>=0 for v in counts),'Invalid request histogram')
    a,b,other,ra,rb,ro,pairs,hits,errors=counts
    require(other==ro==errors==0,'Invalid pair input or transfer size')
    require(sum(counts[:6])==device['completed_commands'],'Command histogram does not reconcile')
    require(512*(a+ra)+1024*(b+rb)==device['completed_bytes'],'Byte histogram does not reconcile')
    require(ra+rb==device['replay_commands'] and 512*ra+1024*rb==device['replay_bytes'],'Replay histogram differs')
    require(hits<=2*pairs,'Invalid pair cache hits')
    if mixed and require_pair:require(b>0 and a>0,'Both actual 512B and 1KiB primary reads required')
    if not mixed:require(b==rb==pairs==hits==0,'Single-row control issued paired reads')


def validate_completion(r):
    p=read(HERE/'protocol.json')
    require(r['protocol']==p and r['variant'] in VARIANTS and r['read_mode']==r['variant'],'Wrong read protocol')
    require(r['mixed_io']==(r['variant']=='mixed') and r['mean_mode']=='legacy','Wrong execution mode')
    # Reuse only the reference's unchanged training/cache/model gates. The
    # changed read-path contract is checked separately above and below.
    reference.validate_completion(dict(r,variant='legacy',mixed_io=False,protocol=read(reference.HERE/'protocol.json')))
    check_pair_counts(r['pair_counts'],r['device_io'],r['mixed_io'])
    return True


def full_schedule():return ['single','mixed','mixed','single']


def compare_workload(a,b):
    for r in (a,b):validate_completion(r)
    for key in ('root_order_sha256','initial_model_sha256','final_model_sha256','losses','shapes',
                'cpu_cache','feature_access','logical_feature_requests','sampler_binary_sha256'):
        require(a[key]==b[key],'Read-path comparison changed workload/numerics: '+key)
    return dict(passed=True,losses_bit_exact=True,final_model_hash_equal=True)
