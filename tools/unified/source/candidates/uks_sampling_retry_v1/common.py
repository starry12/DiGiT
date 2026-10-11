from candidates.uks_sampling_profile_v1.common import *
from candidates.uks_sampling_profile_v1 import common as parent
HERE=ROOT/'candidates/uks_sampling_retry_v1'
OUT=parent.OUT/'retry1'
def verify():
    m=read(HERE/'manifest.json');require(parent.verify()==m['parent_sha256'],'Original diagnostic changed')
    for n,h in m['files'].items():require(sha(HERE/n)==h,'Retry source changed: '+n)
    for n,h in m['evidence'].items():require(sha(ROOT/n)==h,'Prior evidence changed: '+n)
    return sha(HERE/'manifest.json')
def check_ready():
    verify();parent.check_ready();d=parent.OUT/'gpu_check';a=read(d/'accepted.json')
    require(a['passed'] and a['normal_exit'] and read(d/'exit.json')['returncode']==0 and a['source_sha256']==parent.verify() and sha(d/'report.json')==a['report_sha256'],'Prior GPU parity not accepted')
