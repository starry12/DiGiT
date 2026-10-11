"""Same four-arm workload with explicit removal of continuous GPU monitoring."""
import copy
from .common import HERE,LARGE,read,sha,require,ARMS


def compile_plan():
    origin=read(HERE/'preparation_origin.json')
    require(sha(origin['protocol']['path'])==origin['protocol']['sha256'],'Original protocol changed')
    p=copy.deepcopy(read(origin['protocol']['path']))
    p['execution'].update(schema='digit-cache-controller-v5',large_output_root=str(LARGE),
        monitor='disabled',monitor_query_timeout_seconds=None,monitor_period_after_query_seconds=None,
        monitor_max_gap_enforced=False,resource_evidence='CUDA stage checkpoints only',
        stages=['reuse_preparation','full','aggregate'],
        preparation_origin_sha256=sha(HERE/'preparation_origin.json'),
        inherited_short_origin_sha256=sha(HERE/'short_origin.json'),
        device_io_window_maxima='exact if inferable; otherwise null with cumulative upper bounds',
        device_io_epoch_maxima='exact from reset counters')
    p['implementation'].update(native_ready=False,controller_source_ready=True)
    return p


def validate(p):
    require(p==compile_plan(),'Frozen v4 protocol differs: only continuous monitoring and preparation reuse change')
    return p


def schedule(p):
    validate(p)
    return ([dict(stage='reuse_preparation',arm=None)]+
            [dict(stage='full',arm=a) for a in ARMS]+[dict(stage='aggregate',arm=None)])
