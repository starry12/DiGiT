"""Region-scoped useful SSD payload, independent of model or graph."""
import math
NAMES = ('enabled','region_id','ssd_fill_bytes','ssd_useful_bytes','gpu_feature_bytes','gpu_feature_rows')
COUNTERS = NAMES[2:]
SCHEMA = 'digit-useful-io-v1'

def require(ok, message):
    if not ok: raise RuntimeError(message)

def decode(values):
    require(len(values)==6 and all(type(v) is int and v>=0 for v in values), 'Invalid native useful counters')
    result=dict(zip(NAMES,values))
    require(result['enabled'] in (0,1), 'Invalid useful enable flag')
    result['schema']=SCHEMA
    return result

def begin_region(loader):
    loader.BAM_FS.begin_useful_io_region()

def useful_interval(a,b,feature,feature_bytes=512):
    require(a['schema']==b['schema']==SCHEMA and a['enabled']==b['enabled']==1,
            'Missing useful I/O instrumentation')
    require(a['region_id']==b['region_id'] and a['region_id']>0, 'Counter region changed within interval')
    d={k:b[k]-a[k] for k in COUNTERS}
    require(all(v>=0 for v in d.values()), 'Useful counter went backwards')
    require(d['gpu_feature_rows']==feature['gpu_ssd'], 'GPU feature row accounting incomplete')
    require(d['gpu_feature_bytes']==d['gpu_feature_rows']*feature_bytes, 'Wrong logical feature payload width')
    require(d['ssd_useful_bytes']<=d['gpu_feature_bytes'], 'Useful bytes exceed consumed features')
    return dict(schema=SCHEMA,enabled=True,region_id=a['region_id'],reconciled=True,
                granularity_bytes=512,**d)

def validate_region(interval):
    u=interval['useful_io'];d=interval['device']
    require(u['schema']==SCHEMA and u['enabled'] and u['reconciled'] and d['enabled'] and d['reconciled'], 'Unreconciled region')
    for source,keys in ((u,COUNTERS),(d,('completed_bytes','primary_bytes','replay_bytes','submitted_commands','completed_commands','active_ns')),
                        (interval['feature'],('cpu','gpu_ssd'))):
        require(all(type(source[k]) is int and source[k]>=0 for k in keys), 'Invalid region counters')
    width=interval['feature_row_bytes']
    require(width in (512,4096) and u['gpu_feature_rows']==interval['feature']['gpu_ssd'] and
            u['gpu_feature_bytes']==u['gpu_feature_rows']*width, 'Wrong region feature accounting')
    require(u['ssd_useful_bytes']<=u['gpu_feature_bytes'], 'Useful bytes exceed logical payload')
    require(d['completed_bytes']==d['primary_bytes']+d['replay_bytes'], 'Physical bytes mismatch')
    require(d['submitted_commands']==d['completed_commands'], 'Unfinished I/O')
    require(u['ssd_fill_bytes']==d['primary_bytes'], 'Cache fill differs from primary SSD read bytes')
    require(u['ssd_useful_bytes']<=u['ssd_fill_bytes'], 'Useful bytes exceed region SSD reads')
    # Only a complete region can enforce this bound: a window may consume
    # data fetched in a previous window of the same region.
    return True

def summarize(regions):
    keys=('logical_feature_bytes','gpu_feature_bytes','ssd_useful_bytes','ssd_completed_bytes',
          'ssd_primary_bytes','ssd_replay_bytes','ssd_active_seconds','feature_seconds','train_seconds')
    totals={k:0 for k in keys}
    for interval,seconds in regions:
        validate_region(interval)
        require(math.isfinite(seconds) and seconds>0, 'Invalid training duration')
        u=interval['useful_io'];d=interval['device'];f=interval['feature']
        require(math.isfinite(interval['feature_seconds']) and interval['feature_seconds']>=0, 'Invalid feature duration')
        width=interval['feature_row_bytes']
        require(width in (512,4096), 'Unsupported feature width')
        totals['logical_feature_bytes']+=(f['cpu']+f['gpu_ssd'])*width
        totals['gpu_feature_bytes']+=u['gpu_feature_bytes'];totals['ssd_useful_bytes']+=u['ssd_useful_bytes']
        for out,source in (('ssd_completed_bytes','completed_bytes'),('ssd_primary_bytes','primary_bytes'),('ssd_replay_bytes','replay_bytes')):
            totals[out]+=d[source]
        totals['ssd_active_seconds']+=d['active_ns']/1e9
        totals['feature_seconds']+=interval['feature_seconds'];totals['train_seconds']+=seconds
    def ratio(a,b):return totals[a]/totals[b] if totals[b]>0 else None
    result=dict(schema=SCHEMA,region_count=len(regions),**totals)
    for key,numerator,denominator in (
        ('ssd_useful_gbps','ssd_useful_bytes','ssd_active_seconds'),
        ('ssd_physical_gbps','ssd_completed_bytes','ssd_active_seconds'),
        ('ssd_useful_per_train_gbps','ssd_useful_bytes','train_seconds'),
        ('effective_feature_gbps','logical_feature_bytes','feature_seconds')):
        value=ratio(numerator,denominator);result[key]=None if value is None else value/1e9
    result['ssd_payload_utilization']=ratio('ssd_useful_bytes','ssd_completed_bytes')
    result['scope']='sum of complete regions; unique bytes per cache fill consumed in the same region'
    return result

def install_runner(r,feature_bytes=512):
    """Install before importing a loop using from runner import *. No disk edits."""
    require(not getattr(r,'_useful_io_installed',False), 'Accounting already installed')
    old_snapshot,old_interval=r.snapshot,r.interval
    def snapshot(loader):
        result=old_snapshot(loader)
        if 'source_rows' not in result: result['useful_io']=decode(list(loader.BAM_FS.get_useful_io_stats()))
        return result
    def interval(a,b,rows):
        result=old_interval(a,b,rows)
        if not result.get('source_only'):
            result['useful_io']=useful_interval(a['useful_io'],b['useful_io'],result['feature'],feature_bytes)
            result['feature_row_bytes']=feature_bytes
        return result
    r.snapshot=snapshot;r.interval=interval;r._useful_io_installed=True
