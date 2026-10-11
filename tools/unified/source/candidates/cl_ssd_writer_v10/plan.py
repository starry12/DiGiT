"""CL extents: preserve all accepted datasets; reuse only pinned failed UKL v14."""
from candidates.ukl_feature_binding_v13.plan import (ROOT,DEVICE,PAGE,ROW,GIB,digest,ident,
    read_metadata,inventory,align,write_schedule,sample_pages)
ROWS={'gids':978408098,'digit':1174089716}
OFFSETS={'gids':5121*GIB,'digit':3270*GIB}
FAILED_NAMES=('libnvm0.ukl14.offset5498631880704.json','libnvm0.ukl14.offset5902358806528.json')

def failures(snapshot):
    byfile={x['path']:x for x in snapshot['files']}
    found=[]
    for row in snapshot['regions']:
        if row['label'] in [str(ROOT/'ssd_state'/n) for n in FAILED_NAMES]:
            if row['status']!='failed_occupied':raise RuntimeError('Old extent is no longer failed')
            found.append(dict(row,path=row['label'],sha256=byfile[row['label']]['sha256']))
    if len(found)!=2:raise RuntimeError('Exact two failed UKL v14 reservations required')
    return found

def validate_protected(snapshot,arms,exceptions):
    old=failures(snapshot)
    if old!=exceptions:raise RuntimeError('Failed reservations changed')
    excluded={v['label'] for v in old}
    for r in arms.values():
        lo,hi=r['guard_before'][0],r['guard_after'][1]
        if lo<0 or hi>snapshot['capacity_bytes']:raise RuntimeError('Outside device capacity')
        for occupied in snapshot['regions']+snapshot['reservations']:
            if occupied['label'] not in excluded and lo<occupied['end'] and occupied['start']<hi:
                raise RuntimeError('Overlaps protected range: '+occupied['label'])

def make_plan(snapshot):
    clean=dict(snapshot);claimed=clean.pop('sha256')
    if digest(clean)!=claimed:raise RuntimeError('Bad inventory digest')
    arms={}
    for arm,rows in ROWS.items():
        lo=OFFSETS[arm];logical=rows*ROW;payload=align(logical,PAGE);hi=lo+payload
        arms[arm]=dict(device_offset_bytes=lo,end_bytes_exclusive=hi,storage_rows=rows,
            row_bytes=ROW,logical_bytes=logical,payload_bytes=payload,zero_tail_bytes=payload-logical,
            source_header_bytes=0,source_format='raw float32 little-endian',write_alignment_bytes=PAGE,
            guard_before=[lo-PAGE,lo],guard_after=[hi,hi+PAGE])
    validate_protected(snapshot,arms,failures(snapshot))
    return dict(schema='cl-feature-regions-v10',inventory_sha256=claimed,device=DEVICE,
        expected_device=snapshot['expected_device'],arms=arms,raw_ssd_written=False)

def bind_sources(plan,snapshot,accepted):
    if plan!=make_plan(snapshot):raise RuntimeError('CL region plan changed')
    if not accepted.get('prepared') or accepted.get('dataset')!='CL' or accepted['nodes']!=ROWS['gids'] or accepted['storage_rows']!=ROWS['digit']:
        raise RuntimeError('Wrong CL preparation')
    arms={}
    for arm,r in plan['arms'].items():
        stage=accepted['stages']['features-'+arm];source=stage['files'][arm+'_features.f32']
        if source['bytes']!=r['logical_bytes'] or not source['direct_io'] or len(source['sha256'])!=64:
            raise RuntimeError('Invalid CL feature receipt')
        arms[arm]=dict(region=r,source=source,acceptance_sha256=stage['sha256'],
            generation_sha256=source['sha256'],device_sample_plan=sample_pages(r))
    return dict(schema='cl-feature-source-binding-v10',plan_sha256=digest(plan),
        input_index_sha256=digest(accepted),inventory_sha256=snapshot['sha256'],arms=arms,
        sources_bound=True,raw_ssd_bound=False)
