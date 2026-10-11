"""Disjoint extent plans, registry reservation, guarded writer and full readback."""
from dataclasses import asdict
from types import SimpleNamespace
import math,os,subprocess,time
import numpy as np
from .common import *
from .binding import protocol,check

def api():
    from .runtime import setup_sampling_imports
    setup_sampling_imports()
    from digit import ssd_payload
    return ssd_payload

def plan(arm):
    b=check();p=protocol();file=DATA/('synthetic' if arm=='gids' else 'payload')/'features.npy'
    value=api().build_plain_payload_plan(file,page_size=4096,device_offset_bytes=p['ssd_offsets'][arm])
    require(value.feature_file_sha256==b['files'][str(file)]['sha256'],'Feature source mismatch')
    require(value.device_offset_bytes+value.payload_bytes<=read(ROOT/'configs/device.json')['capacity_bytes'],'SSD capacity exceeded')
    return value

def pages_for(total):
    # Require exact chunks. A larger exact divisor avoids millions of tiny writes.
    values=[]
    for k in range(1,math.isqrt(total)+1):
        if total%k==0:values.extend([k,total//k])
    valid=[v for v in values if 4096<=v<=2**21]
    require(bool(valid),'No suitable exact writer cache divisor')
    return min(valid)

def state_path(v):return ROOT/'ssd_state'/('libnvm0.offset'+str(v.device_offset_bytes)+'.json')

def prepare(arm,folder):
    a=api();v=plan(arm);folder.mkdir(exist_ok=True,parents=True)
    from ae.common import check_device
    check_device();path=state_path(v)
    if path.exists():
        s=read(path);a._validate_active_state(path,v,s['status'])
        require(s['status'] in ('written_unverified','verified'),'Partial write preserved; inspect before overwrite')
        require((folder/a.WRITE_RECEIPT).exists(),'State exists without this run write receipt')
        return v
    a.require_unoccupied_active_range(v,ROOT/'ssd_state')
    command=a.build_write_command(v,ROOT/'bam/build/bin/nvm-readwrite_stripe-bench-jc',gpu=0,cache_pages=pages_for(v.payload_pages))
    write(folder/'write_plan.json',dict(asdict(v),command=command))
    write(path,a._active_binding(v,'write_in_progress'))
    with (folder/'writer.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(path,a._active_binding(v,'written_unverified'))
    write(folder/a.WRITE_RECEIPT,dict(asdict(v),status='written_unverified',active_state_file=str(path)))
    return v

def verify_payload(arm,folder):
    from .runtime import prepare_imports
    prepare_imports();a=api();v=plan(arm)
    receipt=folder/a.VERIFY_RECEIPT
    if receipt.exists():
        a._validate_active_state(state_path(v),v,'verified');a.validate_verify_receipt(read(receipt),v);return read(receipt)
    # Frozen full-value readback through the new 256D binary, fresh process.
    a._verify(SimpleNamespace(skip_checksums=False,chunk_rows=65536,cache_size=64,report_dir=str(folder),gpu=0),v)
    return read(receipt)
