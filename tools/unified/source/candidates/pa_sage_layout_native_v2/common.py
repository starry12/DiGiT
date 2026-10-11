"""Fixed, serial PA layout experiment; preview never opens a device."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import time
from ae.common import sha, write, require, host

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/pa_sage_layout_native_20260925_v2'
PLAN=ROOT/'results/pa_sage_layout_sweep_20260924_v1/plan_n0.json'
INDEX=ROOT/'results/pa_sage_layout_perf_20260924_v2/protocols_final/index.json'
PY='/home/embed/miniconda3/envs/gids/bin/python'
UNIT='digit-pa-layout-grid-20260925-v2'
DEVICE='/dev/libnvm0'
OFFSET=4*2**40
SLOT_BYTES=2**40
REGISTRY=ROOT/('ssd_state/libnvm0.offset%d.json'%OFFSET)
GPU_UUID='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'
ORDER=['g2_r20','g1_r00','g1_r80','g4_r00','g4_r80','g2_r00','g2_r80',
       'g1_r10','g1_r20','g1_r40','g2_r10','g2_r40','g4_r10','g4_r20','g4_r40']

def read(path):return json.loads(Path(path).read_text())

def verify():
    from candidates.pa_sage_layout_perf_v3.common import verify as native_verify
    from candidates.pa_sage_layout_sweep_v1.common import code_bindings
    m=read(HERE/'manifest.json')
    require(native_verify()==m['native_candidate_sha256'],'Native candidate changed')
    require(code_bindings()==m['layout_code_bindings'],'Layout builder changed')
    for path,digest in m['files'].items():require(sha(ROOT/path)==digest,'Grid input/code changed: '+path)
    return sha(HERE/'manifest.json')

def cells():
    plan=read(PLAN);index=read(INDEX)
    a={x['id']:x for x in plan['points']};b={x['point']['id']:x for x in index['points']}
    require(len(a)==len(b)==len(ORDER)==15 and set(a)==set(b)==set(ORDER),'Incomplete grid')
    result=[]
    for key in ORDER:
        item=b[key];p=read(item['protocol'])
        require(sha(item['protocol'])==item['protocol_sha256'],'Protocol changed')
        require(p['epochs']==1 and p['evaluation']=='disabled' and p['warmup_batches']==0,'Wrong epoch protocol')
        require(a[key]['filesystem_destination']==p['base_layout'],'Layout/protocol path mismatch')
        require(Path(p['base_layout']).is_relative_to('/mnt/n0') if hasattr(Path,'is_relative_to')
                else str(Path(p['base_layout'])).startswith('/mnt/n0/'),'Large files must use /mnt/n0')
        result.append(dict(point=key,spec=a[key],protocol=item['protocol'],layout=p['base_layout']))
    return result

@contextlib.contextmanager
def device_locks():
    # Same locks as the AE CLI, author native runs and historical SSD provisioner.
    paths=[('/run/digit-ae-selfservice/exclusive.lock','r+'),
           ('/tmp/digit-pa-bidir-controller.lock','a+'),
           ('/tmp/digit-pa-sage-libnvm0.lock','a+'),
           (str(ROOT/'ssd_state/libnvm0.prepare.lock'),'a+')]
    with contextlib.ExitStack() as stack:
        for path,mode in paths:
            f=stack.enter_context(open(path,mode));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def progress(folder,stage,**kw):
    value=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw)
    write(Path(folder)/'progress.json',value);print(json.dumps(value),flush=True)

def writer_pages(payload_bytes):
    """Writer uses float32 ceil(length/chunk); require exact bounded chunks."""
    import math
    import numpy as np
    require(type(payload_bytes) is int and 0<payload_bytes<=SLOT_BYTES and payload_bytes%4096==0,'Invalid payload size')
    pages=payload_bytes//4096
    for cache_pages in range(min(16384,pages),0,-1):
        if pages%cache_pages:continue
        iterations=pages//cache_pages
        if math.ceil(float(np.float32(payload_bytes)/np.float32(cache_pages*4096)))==iterations:
            return cache_pages
    raise RuntimeError('No safe exact writer chunk geometry')

def reject_overlaps(rows,length,allow_registry=False):
    require(0<length<=SLOT_BYTES and length%4096==0,'Invalid scratch extent')
    require(OFFSET+length+4096<=read(ROOT/'configs/device.json')['capacity_bytes'],'SSD capacity exceeded')
    for row in rows:
        if allow_registry and Path(row['path']).resolve()==REGISTRY.resolve():continue
        require(not (row['start']<OFFSET+length+4096 and OFFSET-4096<row['end']),
                'Scratch extent/guards overlap registered data: '+str(row))

def inventory():
    import digit_paths
    roots={ROOT,Path(digit_paths.storage_root()),Path('/home/embed/gids_all'),Path('/home/embed/digit-ae-clean')}
    releases=Path('/srv/digit-ae/releases')
    if releases.exists():roots.update(p for p in releases.iterdir() if p.is_dir())
    paths=set()
    for root in roots:
        for name in ('ssd_state','data/papers/ssd','data/igb/ssd'):
            paths.update((root/name).glob('*.json'))
        if root in (Path('/mnt/n3/gids_all'),Path('/home/embed/gids_all')):
            paths.update((root/'results').glob('*/ssd*receipt.json'))
    rows=[]
    for path in sorted(paths):
        d=read(path)
        if d.get('device')!=DEVICE or 'payload_bytes' not in d:continue
        start=int(d.get('device_offset_bytes',0));length=int(d['payload_bytes'])
        require(start>=0 and length>0,'Invalid registered range: '+str(path))
        rows.append(dict(path=str(path),sha256=sha(path),start=start,end=start+length,status=d.get('status')))
    return rows
