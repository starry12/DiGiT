"""Frozen original runtime and data; separate source and evidence for retry."""
from candidates.uks_native_v1.common import *
from candidates.uks_native_v1 import common as parent
HERE=ROOT/'candidates/uks_native_retry_v1'
OUT=parent.OUT/'retry1'

def verify():
    m=read(HERE/'manifest.json')
    require(parent.verify()==m['parent_source_sha256'],'Original source changed')
    for name,value in m['files'].items():require(sha(HERE/name)==value,'Repair source changed: '+name)
    for name,value in m['evidence'].items():require(sha(ROOT/name)==value,'Reused evidence changed: '+name)
    return sha(HERE/'manifest.json')

def check_reuse():
    verify();parent.binary_receipt()
    from candidates.uks_native_v1.binding import check
    from candidates.uks_native_v1.controller import accept
    check();s=read(parent.OUT/'status.json')
    for stage in ('profile','write_gids','verify_gids','write_digit','verify_digit'):
        folder=parent.OUT/stage;r=read(folder/'accepted.json')
        require(sha(folder/'accepted.json')==s['completed'][stage],'Unaccepted prior stage: '+stage)
        accept(stage,r,read(folder/'exit.json')['returncode'],parent.verify())
        require(r['normal_exit'] and sha(folder/'report.json')==r['report_sha256'],'Prior worker receipt changed')
    return True
