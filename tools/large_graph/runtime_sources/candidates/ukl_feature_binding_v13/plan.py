"""Fail-closed registry inventory and append-only candidate extent plans.

An unregistered range is NOT proof of ownership or blank media. This module
never reserves production state or opens /dev/libnvm0, including for reads.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import stat

ROOT = Path(__file__).resolve().parents[2]
DEVICE = '/dev/libnvm0'
PAGE, ROW, GIB = 4096, 512, 2**30
ROWS = {'gids': 787801471, 'digit': 945361765}
MAX_JSON = 2**21


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def ident(s):
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def _pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate JSON key: '+key)
        result[key]=value
    return result


def read_metadata(path):
    path=Path(path); resolved=path.resolve(strict=True)
    before=resolved.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_JSON:
        raise ValueError('Only bounded regular metadata files: '+str(path))
    with resolved.open('rb') as f: raw=f.read(MAX_JSON+1)
    if len(raw)!=before.st_size or ident(resolved.stat())!=ident(before) or path.resolve(strict=True)!=resolved:
        raise RuntimeError('Metadata changed during snapshot: '+str(path))
    value=json.loads(raw,object_pairs_hook=_pairs)
    if not isinstance(value,dict):raise ValueError('Metadata object required')
    return value,dict(path=str(path),resolved=str(resolved),identity=ident(before),
                      sha256=hashlib.sha256(raw).hexdigest())


def integer(value, minimum=0):
    if type(value) is not int or value<minimum:raise ValueError('Invalid nonnegative integer')
    return value


def extent(value, capacity, label):
    start=integer(value.get('device_offset_bytes',0))
    length=integer(value['payload_bytes'],1)
    if start%ROW or length%ROW or start+length>capacity:
        raise ValueError('Unaligned/out-of-capacity registered extent: '+label)
    # Every status remains occupied, even unknown, stale or failed entries.
    return dict(start=start,end=start+length,label=label,status=value.get('status','unknown'))


def _constant(node):
    if isinstance(node,ast.Constant):return integer(node.value)
    if isinstance(node,ast.BinOp):
        left,right=_constant(node.left),_constant(node.right)
        if isinstance(node.op,ast.Mult):return left*right
        if isinstance(node.op,ast.Pow) and left<=1024 and right<=64:return left**right
    raise ValueError('Unrecognized reserved-slot constant')


def inventory():
    device,devfile=read_metadata(ROOT/'configs/device.json')
    capacity=integer(device['capacity_bytes'],PAGE)
    if device.get('logical_block_bytes')!=ROW or device.get('namespace')!=1:
        raise ValueError('Unexpected device geometry')
    locations,locfile=read_metadata(ROOT/'configs/external_paths.json')
    storage=Path(locations['storage_root'])
    if not storage.is_absolute():raise ValueError('Absolute storage root required')
    roots={ROOT,storage,Path('/home/embed/gids_all'),Path('/home/embed/digit-ae-clean')}
    releases=Path('/srv/digit-ae/releases')
    # iterdir/stat errors must not be mistaken for an empty registry.
    release_names=sorted(p.name for p in releases.iterdir() if p.is_dir())
    roots.update(releases/name for name in release_names)
    paths=set(); directories={}
    for root in sorted(roots):
        for name in ('ssd_state','data/papers/ssd','data/igb/ssd'):
            folder=root/name
            try:
                entries=sorted(p for p in folder.iterdir() if p.suffix=='.json')
            except FileNotFoundError:
                directories[str(folder)]=None;continue
            directories[str(folder)]=[p.name for p in entries];paths.update(entries)
        if root in (storage,Path('/home/embed/gids_all')):
            parent=root/'results'
            try:children=list(parent.iterdir())
            except FileNotFoundError:children=[]
            if len(children)>10000:raise ValueError('Legacy metadata scan bound')
            for child in children:
                if child.is_dir():
                    # Bounded directory scan: never recurse through datasets.
                    paths.update(p for p in child.iterdir() if p.name.startswith('ssd') and p.name.endswith('receipt.json'))
    if len(paths)>10000:raise ValueError('Metadata count bound')
    files=[devfile,locfile];regions=[]
    for path in sorted(paths):
        value,receipt=read_metadata(path);files.append(receipt)
        if value.get('device')==DEVICE:
            if 'payload_bytes' not in value:raise ValueError('Device record lacks extent: '+str(path))
            regions.append(extent(value,capacity,str(path)))
        elif path.parent.name=='ssd_state' and 'device' not in value:
            raise ValueError('State missing device identity: '+str(path))
    # Preserve the entire historical PA layout scratch slot, not just its
    # currently occupied payload. Extract literal arithmetic without imports.
    slotfile=ROOT/'candidates/pa_sage_layout_native_v2/common.py'
    raw=slotfile.read_bytes()
    if len(raw)>MAX_JSON:raise ValueError('Reserved-slot source bound')
    values={}
    for node in ast.parse(raw).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in ('OFFSET','SLOT_BYTES'):values[key]=_constant(node.value)
    reserved=extent(dict(device_offset_bytes=values['OFFSET']-PAGE,
                         payload_bytes=values['SLOT_BYTES']+2*PAGE),capacity,'historical PA layout whole slot plus guards')
    files.append(dict(path=str(slotfile),resolved=str(slotfile.resolve()),identity=ident(slotfile.stat()),sha256=hashlib.sha256(raw).hexdigest()))
    result=dict(device=DEVICE,expected_device=device,device_identity_confirmed=False,
                capacity_bytes=capacity,directories=directories,release_names=release_names,
                files=files,regions=regions,reservations=[reserved],
                scope='known registries, AE releases, legacy receipts and PA scratch reservation',
                raw_device_opened=False)
    result['sha256']=digest(result)
    return result


def align(value,unit):return (value+unit-1)//unit*unit


def make_plan(snapshot):
    clean=dict(snapshot); claimed=clean.pop('sha256')
    if claimed!=digest(clean):raise ValueError('Inventory digest mismatch')
    capacity=integer(snapshot['capacity_bytes'],PAGE)
    protected=snapshot['regions']+snapshot['reservations']
    if not protected:raise ValueError('Empty occupancy inventory is not an allocation basis')
    for r in protected:
        if not 0<=integer(r['start'])<integer(r['end'],1)<=capacity:
            raise ValueError('Invalid protected interval')
    cursor=max(r['end'] for r in protected);arms={}
    for arm,rows in ROWS.items():
        begin=align(cursor+PAGE,GIB);logical=rows*ROW;padded=align(logical,PAGE)
        end=begin+padded
        if end+PAGE>capacity:raise RuntimeError('No append-only capacity for both UKL arms')
        if any(begin-PAGE<r['end'] and r['start']<end+PAGE for r in protected):
            raise RuntimeError('Proposed extent/guards overlaps protected range')
        arms[arm]=dict(device_offset_bytes=begin,end_bytes_exclusive=end,
                       storage_rows=rows,row_bytes=ROW,logical_bytes=logical,
                       payload_bytes=padded,zero_tail_bytes=padded-logical,
                       source_header_bytes=0,source_format='raw float32 little-endian',
                       write_alignment_bytes=PAGE,guard_before=[begin-PAGE,begin],
                       guard_after=[end,end+PAGE])
        cursor=end+PAGE
    return dict(schema='ukl-feature-region-plan-v13',inventory_sha256=claimed,
                device=DEVICE,expected_device=snapshot['expected_device'],arms=arms,
                remaining_tail_bytes=capacity-cursor,registry_reserved=False,
                device_identity_confirmed=False,ownership_proven=False,
                raw_ssd_written=False,raw_ssd_bound=False,
                required_before_writes=['accepted feature receipts','fresh inventory under exclusive locks',
                    'actual namespace identity and capacity','exclusive device idle admission',
                    'new extent ownership evidence or bounded full blank scan','guard hashes before write',
                    'bounded writer and independent writeback/kernel monitoring'])


def validate_plan(plan,snapshot):
    if plan!=make_plan(snapshot):raise RuntimeError('Stale or edited region plan; regenerate and review')


def write_schedule(region,chunk_bytes=8*2**20):
    if type(chunk_bytes) is not int or chunk_bytes<PAGE or chunk_bytes>8*2**20 or chunk_bytes%PAGE:
        raise ValueError('Aligned bounded write chunk required')
    logical=integer(region['logical_bytes'],1);payload=integer(region['payload_bytes'],1)
    base=integer(region['device_offset_bytes'])
    if base%PAGE or payload!=align(logical,PAGE):raise ValueError('Bad region geometry')
    for at in range(0,payload,chunk_bytes):
        size=min(chunk_bytes,payload-at);count=max(0,min(size,logical-at))
        yield dict(source_offset=at,source_bytes=count,device_offset=base+at,
                   write_bytes=size,zero_tail_bytes=size-count)


def sample_pages(region,seed=20261003):
    count=region['payload_bytes']//PAGE
    selected={0,count-1,count//2};rng=random.Random(seed+region['device_offset_bytes'])
    while len(selected)<min(128,count):selected.add(rng.randrange(count))
    return [dict(relative_offset=i*PAGE,device_offset=region['device_offset_bytes']+i*PAGE,
                 source_bytes=max(0,min(PAGE,region['logical_bytes']-i*PAGE))) for i in sorted(selected)]


def bind_sources(plan,snapshot,accepted):
    """Bind accepted file identities to a proposed region, NOT to device contents."""
    validate_plan(plan,snapshot)
    if (accepted.get('prepared') is not True or accepted.get('feature_dim')!=128
            or accepted.get('row_bytes')!=512 or accepted.get('feature_dtype')!='float32'
            or accepted.get('nodes')!=ROWS['gids'] or accepted.get('storage_rows')!=ROWS['digit']):
        raise ValueError('Accepted feature protocol mismatch')
    graph=accepted['stages']['graph']['acceptance_sha256'];arms={}
    for arm in ROWS:
        stage=accepted['stages']['features-'+arm]
        if stage['predecessor']['graph_receipt_sha256']!=graph:raise ValueError('Mixed graph preparations')
        source=stage['files'][arm+'_features.f32'];r=plan['arms'][arm]
        h=source.get('sha256','');p=Path(source['path'])
        if (source.get('bytes')!=r['logical_bytes'] or source.get('direct_io') is not True
                or not p.is_absolute() or p.name!=arm+'_features.f32'
                or len(h)!=64 or any(c not in '0123456789abcdef' for c in h)
                or len(source.get('identity',[]))!=5 or source['identity'][2]!=r['logical_bytes']):
            raise ValueError('Source is not a completed raw feature receipt')
        arms[arm]=dict(region=r,source=source,acceptance_sha256=stage['acceptance_sha256'],
                       generation_sha256=h,device_sample_plan=sample_pages(r),
                       write_chunks=sum(1 for _ in write_schedule(r)))
    return dict(schema='ukl-feature-source-binding-v13',plan_sha256=digest(plan),
                input_index_sha256=digest(accepted),inventory_sha256=snapshot['sha256'],arms=arms,
                sources_bound=True,registry_reserved=False,raw_ssd_written=False,raw_ssd_bound=False,
                device_readback_passed=False,native_training_ready=False)
