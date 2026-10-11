"""Read-only reuse of identified, previously written PA and IG test extents."""
import argparse
import hashlib
import os
import time
import numpy as np
from .common import ROOT,HERE,Path,read,write,sha,require,setup,verify

OUT=ROOT/'results/pa_sage_layout_shared_resume_20260925_v3'
POOL_PROOFS=ROOT/'results/pa_sage_layout_shared_20260925_v1'
POOL_CANDIDATE=ROOT/'candidates/pa_sage_layout_shared_v1/manifest.json'

def identity(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def spec(mode):
    p=read(HERE/'pools.json')[mode]
    require(sha(p['state'])==p['state_sha256'],'Registered payload changed')
    state=read(p['state'])
    require(state['status']=='verified' and state['device']=='/dev/libnvm0' and
            state['device_offset_bytes']==p['offset'] and state['payload_bytes']==p['capacity_bytes'] and
            state['feature_file_sha256']==p['source_full_sha256'],'Wrong pool state')
    require(p['offset']%4096==0 and p['verify_bytes']%4096==0 and 0<p['verify_bytes']<=p['capacity_bytes'],'Invalid pool extent')
    require(identity(p['source'])==p['source_identity'],'Pool source file changed')
    return p

def source_rows(mode):
    p=spec(mode)
    # The IG source is raw float32 despite its .npy filename. The PA source has
    # a real NPY header. Never infer one format from a suffix.
    return np.memmap(p['source'],mode='r',dtype='<f4',offset=p['header_bytes'],shape=(p['capacity_bytes']//512,128))

def check_receipt(mode):
    p=spec(mode);path=POOL_PROOFS/'pools'/(mode+'_receipt.json');r=read(path)
    require(r['candidate_sha256']==sha(POOL_CANDIDATE),'Unexpected pool verification candidate')
    require(r['passed'] and r['mode']==mode and r['raw_ssd_writes'] is False,'Wrong pool receipt')
    require(r['pool_spec_sha256']==sha(HERE/'pools.json') and r['state_sha256']==p['state_sha256'],'Pool binding changed')
    require(r['verified_bytes']==p['verify_bytes'] and r['offset']==p['offset'] and r['all_finite'],'Incomplete pool readback')
    require(r['source_identity']==identity(p['source']),'Source changed since pool readback')
    require(r['actual_sha256']==r['expected_sha256'] and len(r['actual_sha256'])==64,'Readback mismatch')
    return dict(r,receipt_path=str(path))
