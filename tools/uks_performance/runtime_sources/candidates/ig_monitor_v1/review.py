"""Strict, CPU-only review of the controller-owned persistent NVML monitor.

Monitoring errors are evidence of an incomplete observation and always reject
both smoke and full runs. This module never retries or discards raw records.
"""
import math


BACKEND = 'persistent_nvml_ctypes_v1'
MEMORY_API = 'nvmlDeviceGetMemoryInfo_v2'
MEMORY_DEFINITION = 'NVML v2 used bytes excludes driver/firmware reserved memory'
POLICY = {
    'backend': BACKEND,
    'memory_api': MEMORY_API,
    'memory_used_definition': MEMORY_DEFINITION,
    'memory_unit': 'bytes',
    'allowed_error': 'none; every monitoring error rejects smoke and full runs',
    'max_error_fraction': 0.0,
    'max_consecutive_errors': 0,
    'max_sample_gap_seconds': 15.0,
    'sample_gap_comparison': 'strictly_less_than',
    'max_endpoint_gap_seconds': 2.0,
    'max_query_seconds': 5.0,
    'minimum_worker_samples': 10,
    'max_monitor_rss_bytes': 64 * 2**20,
    'scope': 'smoke and full runs; no sparse-timeout qualification',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonnegative_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def assess_monitor(summary, records, ready, worker, controller_pid,
                   resources, returncode, gpu):
    """Reconcile raw observations, process/device identity and worker coverage."""
    limit = POLICY['max_monitor_rss_bytes']
    require(returncode == 0 and summary['complete'] is True
            and summary['sampler_stopped'] is True,
            'Monitor did not finish normally')
    require(ready['passed'] is True and summary['pid'] == ready['pid']
            and summary['parent_pid'] == ready['parent_pid'] == controller_pid,
            'Wrong monitor ownership')
    require(summary['gpu'] == str(gpu) and summary['mode'] == 'external_small_process'
            and summary['backend'] == ready['backend'] == BACKEND,
            'Wrong GPU/monitor backend')
    require(finite(summary['query_timeout_seconds'])
            and 0 < summary['query_timeout_seconds'] <= POLICY['max_query_seconds'],
            'Monitor query timeout exceeds policy')
    require((summary['sampler_shutdown_action'], summary['sampler_returncode'])
            in (('terminated', -15), ('killed', -9)),
            'Sampler did not finish through bounded intentional shutdown')
    sampler_pid = summary['sampler_pid']
    pids = (summary['pid'], sampler_pid, worker['pid'], controller_pid)
    require(all(isinstance(p, int) and not isinstance(p, bool) and p > 0 for p in pids)
            and len(set(pids)) == 4 and ready['sampler_pid'] == sampler_pid,
            'Monitor/sampler not independent or ownership changed')
    gpu_index, gpu_uuid = summary['physical_gpu_index'], summary['physical_gpu_uuid']
    require(nonnegative_integer(gpu_index) and gpu_index == int(gpu)
            and ready['physical_gpu_index'] == gpu_index
            and isinstance(gpu_uuid, str) and gpu_uuid.startswith('GPU-')
            and len(gpu_uuid) > 4 and ready['physical_gpu_uuid'] == gpu_uuid,
            'Wrong physical GPU identity')
    require(resources['worker_pid'] == worker['pid']
            and resources['mode'] == 'checkpoints_only_external_sampler'
            and resources['background_monitor_in_worker'] is False
            and resources['monitor_samples'] == 0
            and resources.get('monitor_error') is None,
            'Wrong worker resources')
    peaks = [summary[name] for name in
             ('peak_rss_bytes', 'supervisor_peak_rss_bytes', 'sampler_peak_rss_bytes')]
    require(all(nonnegative_integer(x) for x in peaks)
            and peaks[0] == peaks[1] + peaks[2] and peaks[0] <= limit,
            'Monitor RSS exceeds budget or combined peak differs')
    require(records and all(finite(x['time_unix']) for x in records),
            'Missing/invalid samples')
    require(all(a['time_unix'] < b['time_unix'] for a, b in zip(records, records[1:])),
            'Nonmonotonic monitor records')
    errors = [x for x in records if 'error' in x]
    good = [x for x in records if 'error' not in x]
    require(errors == summary['errors'] and len(good) == summary['samples'],
            'Monitor summary differs from raw records')
    require(summary['passed'] == (len(good) > 0 and not errors),
            'Wrong original monitor acceptance')
    require(not errors and summary['passed'] is True,
            'Strict monitoring requires zero errors in every phase')
    require(ready['first_sample'] == good[0], 'Readiness sample differs')
    require(all(x['monitor_pid'] == summary['pid']
                and x['monitor_parent_pid'] == controller_pid
                and x['sampler_pid'] == sampler_pid
                and x['sampler_parent_pid'] == summary['pid']
                and x['backend'] == BACKEND
                and x['physical_gpu_index'] == gpu_index
                and x['physical_gpu_uuid'] == gpu_uuid for x in good),
            'Wrong sample process/device identity')
    require([x['sequence'] for x in good] == list(range(1, len(good) + 1)),
            'Missing/duplicate sample sequence')
    for sample in good:
        require(sample['memory_api'] == MEMORY_API
                and sample['memory_used_definition'] == MEMORY_DEFINITION,
                'Wrong device memory API or used-memory definition')
        memory = [sample[name] for name in
                  ('device_total_bytes', 'device_used_bytes', 'device_free_bytes',
                   'device_reserved_bytes')]
        require(all(nonnegative_integer(x) for x in memory) and memory[0] > 0
                and memory[0] == sum(memory[1:]),
                'Invalid device memory byte counters or conservation')
        require(nonnegative_integer(sample['device_used_bytes'])
                and finite(sample['utilization_percent'])
                and 0 <= sample['utilization_percent'] <= 100,
                'Invalid sample metrics')
        rss = [sample[name] for name in
               ('monitor_rss_bytes', 'supervisor_rss_bytes', 'sampler_rss_bytes')]
        require(all(nonnegative_integer(x) for x in rss)
                and rss[0] == rss[1] + rss[2] and rss[0] <= limit
                and rss[1] <= peaks[1] and rss[2] <= peaks[2],
                'Invalid monitor RSS accounting')
        start, end = sample['query_started_unix'], sample['query_finished_unix']
        require(finite(start) and finite(end) and start <= end <= sample['time_unix']
                and end - start <= POLICY['max_query_seconds']
                and sample['time_unix'] - start < POLICY['max_sample_gap_seconds'],
                'Invalid/slow monitor query timing')
    require(all(a['query_finished_unix'] <= b['query_started_unix']
                for a, b in zip(good, good[1:])),
            'Overlapping/nonmonotonic monitor queries')
    began, ended = summary['started_unix'], summary['finished_unix']
    require(finite(began) and finite(ended)
            and began <= good[0]['query_started_unix']
            and good[-1]['time_unix'] <= ended,
            'Samples outside monitor lifetime')
    require(max(x['device_used_bytes'] for x in good) == summary['peak_device_used_bytes'],
            'Monitor peak differs')
    gaps = [b['time_unix'] - a['time_unix'] for a, b in zip(good, good[1:])]
    gap = max(gaps, default=math.inf)
    require(gap < POLICY['max_sample_gap_seconds'], 'Excessive monitoring gap')
    worker_start, worker_end = worker['started_unix'], worker['finished_unix']
    require(finite(worker_start) and finite(worker_end) and worker_start < worker_end
            and good[0]['time_unix'] <= worker_start and ended <= worker_end,
            'Wrong monitor/worker lifetime ordering')
    points = [x for x in good if worker_start <= x['time_unix'] <= worker_end]
    require(len(points) >= POLICY['minimum_worker_samples'], 'Insufficient worker samples')
    checkpoints = resources['checkpoints']
    require(checkpoints and checkpoints[0]['stage'] == 'start'
            and checkpoints[-1]['stage'] == 'training_complete',
            'Incomplete worker checkpoints')
    require(all(finite(x['time_unix'])
                and worker_start <= x['time_unix'] <= worker_end
                and nonnegative_integer(x['device_used_bytes']) for x in checkpoints),
            'Invalid checkpoint or checkpoint outside worker lifetime')
    require(all(a['time_unix'] < b['time_unix']
                for a, b in zip(checkpoints, checkpoints[1:])),
            'Nonmonotonic worker checkpoints')
    endpoint_gap = POLICY['max_endpoint_gap_seconds']
    require(good[0]['time_unix'] <= checkpoints[0]['time_unix'] + endpoint_gap
            and good[-1]['time_unix'] >= checkpoints[-1]['time_unix'] - endpoint_gap,
            'Monitor does not cover training lifetime')
    require(max(x['device_used_bytes'] for x in checkpoints)
            == resources['observed_peak_device_used_bytes'], 'Checkpoint peak differs')
    peak = max(resources['observed_peak_device_used_bytes'],
               max(x['device_used_bytes'] for x in points))
    return dict(
        monitor_pid=summary['pid'], sampler_pid=sampler_pid, worker_pid=worker['pid'],
        backend=BACKEND, physical_gpu_uuid=gpu_uuid, physical_gpu_index=gpu_index,
        memory_api=MEMORY_API, memory_used_definition=MEMORY_DEFINITION, memory_unit='bytes',
        sample_count=len(points), monitor_peak_rss_bytes=peaks[0],
        observed_peak_device_used_bytes=peak, strict_monitor_passed=True,
        qualification='no_query_errors', timeout_count=0, error_count=0,
        query_error_fraction=0.0, max_sample_gap_seconds=gap,
        review_policy=dict(POLICY),
        limitation='Sampled peak is a lower bound; neither finite sampling nor CUDA '
                   'checkpoints establish the unobserved continuous memory peak. '
                   'All monitor errors and gaps of 15 seconds or more reject the run.')
