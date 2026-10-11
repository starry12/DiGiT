"""Policy-aware counters. Never relabel an allocator cache as a feature cache."""
from candidates.pa_sage_cache_policy_v1.metrics import serving
from candidates.uks_native_v1.accounting import decode, useful_interval, validate_region
from candidates.uks_native_v1.counter_math import device_io_interval, gpu_cache_interval
from .common import require, SCRATCH_BYTES

POLICY = ('schema', 'mode', 'state', 'storage_rows', 'cpu_rows', 'gpu_feature_bytes',
          'preload_scratch_bytes', 'preload_rows', 'bypass_ssd_rows', 'address_errors')
GPU = ('policy_id', 'capacity_pages', 'requests', 'hits', 'inserts', 'evictions',
       'resident_pages', 'fifo_ticket', 'partial_fills', 'physical_inserts')
DEVICE = ('enabled', 'outstanding', 'submitted_commands', 'completed_commands',
          'completed_bytes', 'active_ns', 'total_latency_ns', 'max_latency_ns',
          'max_outstanding', 'replay_commands', 'replay_bytes')


def record(names, values):
    require(len(names) == len(values) and all(type(v) is int and v >= 0 for v in values),
            'Malformed native counter vector')
    return dict(zip(names, values))


def snapshot(fs):
    # All native getters synchronize the device. Call only at agreed boundaries.
    policy = record(POLICY, list(fs.policy_stats()))
    require(policy['schema'] == 1 and policy['mode'] in (1, 2, 3) and policy['state'] == 3,
            'Native policy not configured')
    require(policy['address_errors'] == 0, 'Invalid native storage address/CPU slot')
    feature = record(('cpu', 'gpu_ssd'), list(fs.get_feature_access_stats()))
    gpu = record(GPU, list(fs.get_gpu_cache_stats()))
    expected=0 if policy['mode']==3 else 1
    require(gpu.pop('policy_id')==expected,'Wrong allocator')
    gpu['policy']='legacy' if expected==0 else 'fifo'
    device = record(DEVICE, list(fs.get_device_io_stats()))
    require(device['enabled'] == 1 and device['outstanding'] == 0, 'Counters disabled or I/O in flight')
    device['enabled'] = True
    sizes=record(('primary_1k','primary_2k','other_primary','pair_requests','pair_gpu_hits'),list(fs.request_size_stats()))
    return dict(request_sizes=sizes,policy=policy, feature=feature, gpu=gpu, device=device,
                useful_io=decode(list(fs.get_useful_io_stats())))


def interval(before, after, logical_rows, complete_region=False):
    a, b = before['policy'], after['policy']
    for key in POLICY[:-2]:
        require(a[key] == b[key], 'Policy changed during measurement: ' + key)
    require(a['schema'] == 1 and a['state'] == 3 and a['mode'] in (1, 2, 3) and
            a['address_errors'] == b['address_errors'] == 0, 'Invalid policy state/address')
    require(a['preload_rows'] == a['cpu_rows'] and a['preload_scratch_bytes'] == SCRATCH_BYTES,
            'Incomplete preload or wrong scratch allocation')
    feature = {k: after['feature'][k] - before['feature'][k] for k in ('cpu', 'gpu_ssd')}
    require(all(type(v) is int and v >= 0 for v in feature.values()), 'Feature counter reset')
    require(a['cpu_rows'] > 0 or feature['cpu'] == 0, 'Zero-capacity point served CPU feature rows')
    require(before['gpu']['policy']==('legacy' if a['mode']==3 else 'fifo'),'Allocator mode mismatch')
    gpu = gpu_cache_interval(before['gpu'], after['gpu'])
    device = device_io_interval(before['device'], after['device'], complete_region=complete_region)
    require(device['enabled'], 'Missing physical I/O accounting')
    useful = useful_interval(before['useful_io'], after['useful_io'], feature, feature_bytes=1024)
    sizes={k:after['request_sizes'][k]-before['request_sizes'][k] for k in before['request_sizes']}
    require(all(v>=0 for v in sizes.values()) and sizes['other_primary']==0,'Invalid request sizes')
    bypass_rows = b['bypass_ssd_rows'] - a['bypass_ssd_rows']
    require(bypass_rows >= 0, 'Bypass counter reset')
    if a['mode'] == 1:
        require(a['gpu_feature_bytes'] == 0 and
                before['gpu']['capacity_pages'] * 1024 == SCRATCH_BYTES,
                'Static policy secretly allocated persistent GPU features')
        for s in (before['gpu'], after['gpu']):
            require(all(s[k] == 0 for k in GPU[2:]), 'Static path used persistent GPU cache')
        hits = 0
        require(bypass_rows == feature['gpu_ssd'] == device['primary_commands'],
                'Bypass cold requests did not each issue one direct read')
        require(useful['ssd_useful_bytes'] == bypass_rows * 1024,
                'Bypass useful-byte accounting differs from consumed rows')
    else:
        require(a['gpu_feature_bytes'] == before['gpu']['capacity_pages'] * 1024 and
                a['gpu_feature_bytes'] >= SCRATCH_BYTES, 'FIFO capacity differs')
        require(bypass_rows == 0 and gpu['requests'] == feature['gpu_ssd'], 'FIFO route mismatch')
        require(gpu['partial_fills'] == 0 and gpu['inserts'] == sizes['primary_1k']+2*sizes['primary_2k'],
                'Full-page FIFO fill/physical command mismatch')
        hits = gpu['hits']
    require(device['primary_commands']==sizes['primary_1k']+sizes['primary_2k'] and
            device['primary_bytes'] == sizes['primary_1k']*1024+sizes['primary_2k']*2048 and
            useful['ssd_fill_bytes'] == device['primary_bytes'], 'Primary read bytes mismatch')
    result = dict(request_sizes=sizes,pair_dma_staging_bytes=SCRATCH_BYTES,schema='digit-policy-native-interval-v2', policy={1:'bypass',2:'fifo',3:'legacy'}[a['mode']],
        cpu_rows=a['cpu_rows'], preload_rows=a['preload_rows'],
        serving=serving(logical_rows, feature['cpu'], hits, feature['gpu_ssd'] - hits, feature['gpu_ssd']),
        feature=feature, device=device, useful_io=useful, feature_row_bytes=1024,
        gpu_feature_cache_bytes=a['gpu_feature_bytes'],
        gpu_transport_allocation_bytes=before['gpu']['capacity_pages'] * 1024,
        raw_allocator_cache=gpu, bypass_ssd_rows=bypass_rows, reconciled=True,
        complete_region=bool(complete_region))
    # A sub-window can consume a previous window's fills. Enforce useful <= fill
    # only across the full epoch region, whose beginning follows CPU preload.
    if complete_region:
        validate_region(result)
    return result
