"""Same four-arm workload with explicit removal of continuous GPU monitoring."""
import copy
from .common import HERE,LARGE,read,sha,require,ARMS


def compile_plan():
    origin=read(HERE/'preparation_origin.json')
    require(sha(origin['protocol']['path'])==origin['protocol']['sha256'],'Original protocol changed')
    p=copy.deepcopy(read(origin['protocol']['path']))
    p['execution'].update(schema='digit-cache-controller-v6',large_output_root=str(LARGE),
        monitor='disabled',monitor_query_timeout_seconds=None,monitor_period_after_query_seconds=None,
        monitor_max_gap_enforced=False,resource_evidence='CUDA stage checkpoints only',
        stages=['reuse_preparation','smoke','full','aggregate'],
        preparation_origin_sha256=sha(HERE/'preparation_origin.json'),
        device_io_window_maxima='exact if inferable; otherwise null with cumulative upper bounds',
        device_io_epoch_maxima='exact from reset counters')
    p['policy_decision']='equal_CPU_GPU_budgets_legacy_vs_fifo'
    p['capacity_note']='All arms: 11105992 CPU rows (5686267904 bytes) and 4 GiB GPU feature cache. Metadata is separate.'
    for arm,config in p['arms'].items():
        config.update(gpu_policy='fifo' if arm=='digit' else 'legacy',
                      gpu_feature_cache_bytes=4*2**30,lookup_order=['gpu','cpu','ssd'])
    p['execution'].update(backend='pa_sage_cache_policy_v6')
    p['execution'].pop('backend_source_sha256',None)
    p['implementation'].update(native_ready=False,controller_source_ready=True)
    return p


def validate(p):
    require(p==compile_plan(),'Frozen equal-capacity v6 protocol differs')
    return p


def schedule(p):
    validate(p)
    return ([dict(stage='reuse_preparation',arm=None)]+
            [dict(stage='smoke',arm=a) for a in ARMS]+
            [dict(stage='full',arm=a) for a in ARMS]+[dict(stage='aggregate',arm=None)])
