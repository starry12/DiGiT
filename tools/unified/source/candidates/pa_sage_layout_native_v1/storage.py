"""A new SSD slot, blank-checked before first use and recycled only after acceptance.

Archived per-point receipts describe the run-time contents. The live registry is
checked separately by the v3 training gate, so an evicted point cannot run stale.
"""
import argparse
from dataclasses import asdict
import hashlib
import subprocess
import sys
from types import SimpleNamespace
from .common import *

WRITER=ROOT/'bam/build/bin/nvm-readwrite_stripe-bench-jc'

def api():
    from candidates.pa_sage_bidir_native_v2.common import setup
    setup()
    from digit import ssd_payload
    return ssd_payload

def point_paths(key):
    cell=next(x for x in cells() if x['point']==key)
    base=Path(cell['layout']);evidence=OUT/'provision'/key
    return base,evidence

def child(action,key):
    _,folder=point_paths(key)
    with (folder/(action+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-B','-u','-m','candidates.pa_sage_layout_native_v1.storage',action,'--point',key],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def scan(key,after=False):
    """Check every newly allocated byte; never infer free space from no registry."""
    from ae.common import check_device
    check_device();base,folder=point_paths(key);intent=read(folder/'intent.json')
    size=int(intent['plan']['payload_bytes']);blank_from=int(intent['previous_high_water_bytes'])
    sp=api();sp.require_root()
    import torch
    from GIDS import GIDS
    torch.cuda.set_device(0)
    loader=GIDS(page_size=4096,off=OFFSET-4096,num_ele=(size+8192)//4,
        num_ssd=1,cache_size=64,cache_dim=128,ssd_list=[0],ctrl_idx=0,
        feature_index_mode='explicit',mixed_io=False)
    def fetch(first,count):
        ids=torch.arange(first,first+count,device='cuda',dtype=torch.int64)
        values=torch.empty((count,128),device='cuda',dtype=torch.float32)
        loader.BAM_FS.read_feature(values.data_ptr(),ids.data_ptr(),count,128,128,0,0)
        torch.cuda.synchronize();return values.view(torch.uint8)
    guards={name:hashlib.sha256(fetch(first,8).cpu().numpy().tobytes()).hexdigest()
            for name,first in (('before',0),('after',8+size//512))}
    if after:
        require(guards==read(folder/'before.json')['guard_sha256'],'Adjacent SSD pages changed')
        write(folder/'guard.json',dict(passed=True,guard_sha256=guards));return
    checked=0
    for first in range(blank_from//512,size//512,65536):
        count=min(65536,size//512-first);data=fetch(8+first,count)
        if torch.any(data!=0).item():
            write(folder/'blank_failure.json',dict(passed=False,offset=OFFSET+first*512,
                checked_bytes=checked,raw_ssd_write=False,reason='Unregistered nonzero data; ownership must be resolved before overwrite'))
            raise RuntimeError('Unregistered nonzero data in new scratch extent; no write performed')
        checked+=count*512
        if checked%(2**30)==0 or first+count==size//512:
            progress(folder,'blank_scan',checked_bytes=checked,new_bytes=max(0,size-blank_from))
    write(folder/'before.json',dict(passed=True,guard_sha256=guards,
          blank_checked_bytes=checked,previous_owned_prefix_bytes=blank_from,
          offset=OFFSET,bytes=size,raw_ssd_write=False))

def verify_payload(key):
    from ae.common import check_device
    check_device();base,folder=point_paths(key);sp=api()
    plan=sp.build_payload_plan(base/'final/bundle',device_offset_bytes=OFFSET)
    require(asdict(plan)==read(folder/'intent.json')['plan'],'Payload plan changed')
    sp._verify(SimpleNamespace(skip_checksums=False,chunk_rows=65536,cache_size=64,
               gpu=0,report_dir=str(folder)),plan)

def previous_slot():
    slotfile=OUT/'slot.json'
    if not REGISTRY.exists():
        require(not slotfile.exists(),'Live scratch registry disappeared');return None
    require(slotfile.exists(),'Scratch registry is not owned by this experiment')
    slot=read(slotfile)
    require(slot['passed'] and slot['registry_sha256']==sha(REGISTRY),'Incomplete/changed previous provisioning')
    previous=OUT/'native'/slot['point']
    status=read(previous/'status.json');summary=read(previous/'full_summary.json')
    require(status['passed'] and status['complete'] and status['stage']=='complete' and summary['passed'],
            'Previous point has not passed; refuse slot reuse')
    report=previous/'full_digit_full_accepted.json'
    require(summary['report_sha256']['digit_full']==sha(report),'Previous accepted report changed')
    return slot

def provision(key):
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root on physical GPU 2 required')
    with device_locks():
        verify();base,folder=point_paths(key);folder.mkdir(parents=True,exist_ok=False)
        sp=api();from ae.common import check_device
        check_device();previous=previous_slot()
        plan=sp.build_payload_plan(base/'final/bundle',device_offset_bytes=OFFSET)
        size=plan.payload_bytes;rows=inventory();reject_overlaps(rows,size,allow_registry=previous is not None)
        pages=writer_pages(size);command=sp.build_write_command(plan,WRITER,gpu=0,cache_pages=pages)
        intent=dict(point=key,plan=asdict(plan),command=command,registry_ranges=rows,
            previous_slot=previous,previous_high_water_bytes=previous['high_water_bytes'] if previous else 0,
            native_candidate_sha256=sha(ROOT/'candidates/pa_sage_layout_perf_v3/manifest.json'))
        write(folder/'intent.json',intent);write(folder/'device_identity.json',dict(passed=True,expected=read(ROOT/'configs/device.json')))
        progress(folder,'checking_new_extent_and_boundaries',bytes=size);child('before',key)
        require(read(folder/'before.json')['passed'],'Missing blank/boundary check')
        check_device();verify()
        reject_overlaps(inventory(),size,allow_registry=previous is not None)
        require(previous_slot()==previous,'Previous slot changed before write')
        state=folder/'state.json'
        live=sp._active_binding(plan,'write_in_progress');write(state,live);write(REGISTRY,live)
        progress(folder,'writing',bytes=size,offset=OFFSET)
        start=time.time()
        with (folder/'writer.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        live=sp._active_binding(plan,'written_unverified');write(state,live);write(REGISTRY,live)
        write(folder/sp.WRITE_RECEIPT,dict(asdict(plan),status='written_unverified',
            elapsed_seconds=time.time()-start,command=command,active_state_file=str(state)))
        progress(folder,'full_raw_readback',bytes=size);child('verify',key)
        receipt=sp.validate_verify_receipt(folder/sp.VERIFY_RECEIPT,plan)
        require(receipt['verified_bytes']==size and receipt['verified_rows']==plan.storage_rows,'Incomplete raw readback')
        child('guard',key);require(read(folder/'guard.json')['passed'],'Missing boundary check')
        # Keep point evidence immutable, and publish one authoritative live occupant.
        write(REGISTRY,read(state))
        ready=dict(schema='digit-layout-scratch-ready-v1',passed=True,full_readback_passed=True,
            device=DEVICE,offset=OFFSET,bytes=size,file_sha256=plan.feature_file_sha256,
            manifest_sha256=sha(base/'final/bundle/manifest.json'),state=str(state),state_sha256=sha(state),
            verify_receipt=str(folder/sp.VERIFY_RECEIPT),verify_receipt_sha256=sha(folder/sp.VERIFY_RECEIPT),
            active_registry=str(REGISTRY),active_registry_sha256=sha(REGISTRY),
            residency='Live only while active_registry matches; earlier point receipts are historical after slot reuse')
        write(base/'ssd_ready.json',ready)
        write(OUT/'slot.json',dict(passed=True,point=key,registry_sha256=sha(REGISTRY),
            high_water_bytes=max(size,intent['previous_high_water_bytes']),ready_sha256=sha(base/'ssd_ready.json')))
        progress(folder,'complete',passed=True,bytes=size)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('provision','before','verify','guard'));p.add_argument('--point',choices=ORDER,required=True);a=p.parse_args()
    require(os.geteuid()==0,'Native SSD access requires root');verify()
    if a.action=='provision':provision(a.point)
    elif a.action=='verify':verify_payload(a.point)
    else:scan(a.point,after=a.action=='guard')
if __name__=='__main__':main()
