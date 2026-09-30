"""Offline validators for future reports; CPU fixtures cannot pass as native evidence."""
import math
from candidates.pa_sage_cache_policy_v1.common import ARMS, require
from candidates.io_accounting_v1.accounting import validate_region
from .process import validate_evidence
from candidates.pa_sage_cache_policy_v1.metrics import serving

TRACE_KEYS = ('graph_sha256', 'layout_sha256', 'order_sha256', 'sample_trace_sha256',
              'storage_trace_sha256', 'initial_model_sha256', 'feature_receipt_sha256')


def check_serving(region):
    counts=region['serving'];feature=region['feature']
    rebuilt=serving(counts['logical_requests'],feature['cpu'],counts['gpu_hit_rows'],
                    counts['ssd_served_rows'],feature['gpu_ssd'])
    require(rebuilt==counts,'Serving counters differ from native feature counters')


def accept_arm(report, arm, source_sha256, protocol_sha256, short_receipt_sha256, expected_policy):
    require(report['kind'] == 'native_cache_policy_full_epoch' and report['native'] is True and
            report['source_only'] is False and report['worker_returncode'] == 0,
            'Need normal native completion, not a CPU/preflight report')
    require(report['arm'] == arm and report['source_sha256'] == source_sha256 and
            report['protocol_sha256'] == protocol_sha256 and
            report['native_short_receipt_sha256'] == short_receipt_sha256 and
            bool(short_receipt_sha256), 'Changed/unvalidated worker inputs')
    require(report['epochs'] == 1 and report['updates'] == 1179 and report['evaluation'] == 'disabled' and
            report['finite_loss_and_gradients'] is True, 'Incomplete/invalid performance epoch')
    require(math.isfinite(report['training_seconds']) and report['training_seconds'] > 0, 'Invalid training time')
    validate_evidence(report)
    require(len(report['windows']) > 0 and sum(w['updates'] for w in report['windows']) == 1179,
            'Missing measurement windows')
    region = report['region']
    require(region['reconciled'] and region['complete_region'], 'Need complete counter region')
    require(region['policy'] == ('fifo' if arm == 'digit' else 'legacy'), 'Wrong measured policy')
    require(report['cpu_rows'] == expected_policy['cpu_rows'] and
            region['gpu_feature_cache_bytes'] == expected_policy['gpu_feature_cache_bytes'] and
            region['policy'] == expected_policy['gpu_policy'], 'Measured capacities differ from the frozen protocol')
    validate_region(region)
    check_serving(region)
    for key in ('logical_requests', 'cpu_served_rows', 'gpu_hit_rows', 'ssd_served_rows'):
        require(sum(w['region']['serving'][key] for w in report['windows']) == region['serving'][key],
                'Window/epoch mismatch: ' + key)
    for key in TRACE_KEYS:
        require(isinstance(report[key], str) and len(report[key]) == 64, 'Missing input/trace digest: ' + key)
    return report


def compare(reports):
    """Call after accept_arm on each report; no claims about accuracy or convergence."""
    require(set(reports) == set(ARMS), 'Four-arm comparison is incomplete')
    first = reports['degree']
    for report in reports.values():
        for key in TRACE_KEYS:
            require(report[key] == first[key], 'Arms differ in fixed workload: ' + key)
        require(report['cpu_rows'] == first['cpu_rows'], 'CPU capacities differ')
        require(report['region']['gpu_feature_cache_bytes'] == 4*2**30, 'GPU capacities differ')
    require(reports['freq']['hot_nodes_sha256'] == reports['digit']['hot_nodes_sha256'],
            'Freq and DiGiT used different static hot sets')
    return dict(passed=True, workload_matched=True, same_total_cache_capacity=True,
        precision_claim=False, repeats=1,
        arms={name: dict(training_seconds=r['training_seconds'], serving=r['region']['serving'],
                        cpu_feature_bytes=r['cpu_rows'] * 512,
                        gpu_feature_cache_bytes=r['region']['gpu_feature_cache_bytes'])
              for name, r in reports.items()})
