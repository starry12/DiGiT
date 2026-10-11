"""A mutually exclusive serving partition; physical NVMe commands stay separate."""
from .common import require


def serving(logical_requests, cpu_rows, gpu_hit_rows, ssd_rows, gpu_ssd_route_rows):
    counts = [logical_requests, cpu_rows, gpu_hit_rows, ssd_rows, gpu_ssd_route_rows]
    require(all(type(v) is int and v >= 0 for v in counts), 'Invalid request counters')
    require(logical_requests > 0 and cpu_rows + gpu_hit_rows + ssd_rows == logical_requests,
            'Missing or double-counted logical requests')
    require(gpu_ssd_route_rows == gpu_hit_rows + ssd_rows, 'GPU/SSD route differs from its serving partition')
    return dict(logical_requests=logical_requests, cpu_served_rows=cpu_rows, gpu_hit_rows=gpu_hit_rows,
                ssd_served_rows=ssd_rows, gpu_ssd_route_rows=gpu_ssd_route_rows,
                cpu_hit_ratio=cpu_rows / logical_requests, gpu_hit_ratio=gpu_hit_rows / logical_requests,
                combined_hit_ratio=(cpu_rows + gpu_hit_rows) / logical_requests,
                ssd_request_fraction=ssd_rows / logical_requests,
                conditional_gpu_route_hit_ratio=gpu_hit_rows / gpu_ssd_route_rows if gpu_ssd_route_rows else None,
                denominator='all logical feature row requests; three mutually exclusive serving routes')


def from_native_report(report):
    require(report['passed'] and not report['smoke'] and not report['source_only'], 'Need an accepted native full report')
    require(len(report['epochs']) == 1 and report['test'] is None and report['validation_seconds'] == 0,
            'Wrong one-epoch performance report')
    epoch = report['epochs'][0]
    region = epoch['training']
    rows = sum(s['input_nodes'] for s in epoch['shapes'])
    require(epoch['updates'] == len(epoch['shapes']) == report['updates'], 'Incomplete update accounting')
    require(region['gpu']['requests'] == region['feature']['gpu_ssd'], 'Native GPU request accounting differs')
    require(region['gpu']['reconciled'] and region['reconciliation']['reconciled'] and
            region['device']['reconciled'], 'Unreconciled native counters')
    for key in ('cpu', 'gpu_ssd'):
        require(sum(w['feature'][key] for w in epoch['windows']) == region['feature'][key], 'Window/epoch route mismatch')
    require(sum(w['gpu']['hits'] for w in epoch['windows']) == region['gpu']['hits'], 'Window/epoch hit mismatch')
    gpu = region['gpu']['hits']
    result = serving(rows, region['feature']['cpu'], gpu,
                     region['feature']['gpu_ssd'] - gpu, region['feature']['gpu_ssd'])
    io = report['io_accounting_training']
    require(io['logical_feature_bytes'] == rows * 512 and
            io['ssd_completed_bytes'] == io['ssd_primary_bytes'] + io['ssd_replay_bytes'], 'Native byte accounting differs')
    result.update(source='accepted_native_training', training_seconds=epoch['train_seconds'],
        order_excluded_seconds=epoch['train_seconds'] - epoch['order_seconds'],
        physical_io={key: io[key] for key in ('ssd_primary_bytes', 'ssd_completed_bytes', 'ssd_replay_bytes', 'ssd_useful_bytes')},
        cpu_feature_bytes=report['cpu_cache_rows'] * 512, gpu_feature_cache_bytes=report['cache_bytes'])
    return result
