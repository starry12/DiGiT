from candidates.uks_revpr_diagnostic_v1.common import *
from candidates.uks_revpr_diagnostic_v1 import common as parent
HERE=ROOT/'candidates/uks_sampling_profile_v1'
OUT=ROOT/'results/uks_sampling_profile_20260929_v1'
BINARY=HERE/'runtime/UKSSamplingProbeCUDA.so'
def verify():
    m=read(HERE/'manifest.json');require(parent.verify()==m['parent_sha256'],'Parent changed')
    for k,v in m['files'].items():require(sha(HERE/k)==v,'Profile source changed: '+k)
    for k,v in m['dependencies'].items():require(sha(ROOT/k)==v,'Profile dependency changed: '+k)
    return sha(HERE/'manifest.json')
def check_ready():
    verify();parent.check_ready();r=read(HERE/'runtime/build_receipt.json')
    require(r['source_sha256']==verify() and sha(BINARY)==r['binary_sha256'],'Probe build changed')
    require(read(parent.OUT/'summary.json')['passed'],'Prior diagnostic incomplete')
def extension():
    import importlib.util,sys
    name='UKSSamplingProbeCUDA'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,str(BINARY));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]
