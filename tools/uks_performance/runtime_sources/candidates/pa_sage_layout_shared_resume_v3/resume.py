"""Reuse completed pool proofs without disturbing any active service."""
import subprocess,time
from .common import ROOT,read,write,sha,require
from .pool import OUT,POOL_PROOFS,POOL_CANDIDATE,check_receipt

UNITS=('digit-pa-layout-shared-20260925-v1.service','digit-pa-layout-shared-fast-20260925-v2.service')

def prerequisites():
    previous={}
    for unit in UNITS:
        text=subprocess.check_output(['/usr/bin/systemctl','show',unit,'--property=ActiveState,MainPID'],text=True,timeout=30)
        d=dict(line.split('=',1) for line in text.splitlines() if '=' in line)
        require(d['ActiveState'] in ('inactive','failed') and int(d['MainPID'])==0,'Previous service still active: '+unit)
        previous[unit]=d
    state=read(POOL_PROOFS/'status.json')
    require(state['candidate_sha256']==sha(POOL_CANDIDATE) and state['stage']=='failed' and not state['completed'],'Unexpected earlier run state')
    pilot=read(POOL_PROOFS/'calibration/real_first/status.json')
    require(pilot['stage']=='failed' and pilot['workers']==[],'Previous pilot has worker results; review before rerun')
    receipts={mode:check_receipt(mode) for mode in ('real','shared')}
    return previous,receipts

def reuse():
    previous,receipts=prerequisites()
    write(OUT/'pool_reuse.json',dict(passed=True,checked_unix=time.time(),previous_services=previous,
        pool_receipt_sha256={m:sha(r['receipt_path']) for m,r in receipts.items()},
        verified_bytes={m:r['verified_bytes'] for m,r in receipts.items()},raw_ssd_writes=False,new_full_pool_readback=False))
