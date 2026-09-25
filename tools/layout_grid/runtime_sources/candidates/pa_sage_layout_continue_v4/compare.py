"""Post-measurement correction: compare logical requests, report cache routes."""
import argparse
import math
import statistics
from .common import (ROOT, HERE, SOURCE, OUT, NATIVE_SHA, PILOTS,
                     Path, read, write, sha, require, verify)
from candidates.pa_sage_layout_shared_resume_v3.compare import (
    TIME_TOLERANCE, PHYSICAL_IO_TOLERANCE)


def relative(a, b):
    require(math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0,
            'Nonpositive/nonfinite comparison metric')
    return abs(a / b - 1.)


def requests(interval, rows):
    """CPU fallback and GPU/SSD are two serving routes for the same requests."""
    f = interval['feature']
    require(set(f) == {'cpu', 'gpu_ssd'} and
            all(type(v) is int and v >= 0 for v in f.values()), 'Invalid route counter')
    require(sum(f.values()) == rows, 'Logical feature request count differs')
    require(interval['gpu']['requests'] == f['gpu_ssd'], 'GPU route accounting differs')
    require(interval['useful_io']['gpu_feature_rows'] == f['gpu_ssd'], 'Useful route accounting differs')
    require(interval['feature_row_bytes'] == 512, 'Feature row size changed')


def evaluate(reports, smokes):
    require(len(reports) == len(smokes) == 4, 'Need all four ABBA positions')
    require([r['feature_mode'] for r in reports] == ['real', 'shared', 'shared', 'real'],
            'Wrong ABBA order')
    first = reports[0]
    exact = ('initial_parameters_sha256', 'initial_dgl_rng', 'point',
             'layout_manifest_sha256', 'overlay_receipt_sha256', 'policy',
             'candidate_sha256', 'verification_policy', 'cpu_cache_rows', 'cache_bytes')
    routes = []
    for r in reports:
        require(r['passed'] and not r['smoke'] and not r['source_only'] and
                len(r['epochs']) == 1 and r['test'] is None, 'Unaccepted/wrong full run')
        for key in exact:
            require(r[key] == first[key], 'Unpaired pilot field: ' + key)
        a, b = r['epochs'][0], first['epochs'][0]
        require(a['updates'] == r['updates'] == len(a['shapes']) == 1179, 'Incomplete epoch')
        for key in ('updates', 'root_sha256', 'shapes'):
            require(a[key] == b[key], 'Different training sampling: ' + key)
        require(len(a['windows']) == len(b['windows']), 'Different sampling window counts')
        cursor = 1
        for x, y in zip(a['windows'], b['windows']):
            for key in ('start_update', 'end_update', 'input_nodes', 'target_edges',
                        'actual_edges', 'underfilled_owners', 'shortfall_edges', 'group_edges'):
                require(x[key] == y[key], 'Sampling/window count differs: ' + key)
            end = x['end_update']
            require(x['start_update'] == cursor and cursor <= end <= a['updates'],
                    'Sampling windows do not cover a contiguous epoch')
            expected = sum(s['input_nodes'] for s in a['shapes'][cursor - 1:end])
            require(x['input_nodes'] == expected, 'Window request/shape count differs')
            requests(x, expected)
            cursor = end + 1
        require(cursor == a['updates'] + 1, 'Missing final sampling window')
        total = sum(s['input_nodes'] for s in a['shapes'])
        requests(a['training'], total)
        for key in ('cpu', 'gpu_ssd'):
            require(sum(w['feature'][key] for w in a['windows']) == a['training']['feature'][key],
                    'Window/epoch route counter reconciliation differs')
        require(r['io_accounting_training']['logical_feature_bytes'] == total * 512,
                'Logical byte accounting differs')
        routes.append(dict(mode=r['feature_mode'], total=total, **a['training']['feature']))
    for s in smokes:
        require(s['passed'] and s['smoke'] and len(s['audits']) == 4, 'Missing accepted smoke audits')
        for a, b in zip(s['audits'], smokes[0]['audits']):
            for key in ('inputs', 'outputs', 'blocks', 'storage_rows', 'storage_flags', 'cuda_rng', 'dgl_rng'):
                require(a[key] == b[key], 'Different smoke sampling/request trace: ' + key)
    times = {m: [r['epochs'][0]['train_seconds'] - r['epochs'][0]['order_seconds']
                 for r in reports if r['feature_mode'] == m] for m in ('real', 'shared')}
    medians = {m: statistics.median(v) for m, v in times.items()}
    timing_delta = relative(medians['shared'], medians['real'])
    spreads = {m: relative(max(v), min(v)) for m, v in times.items()}
    io_deltas = [relative(r['io_accounting_training'][key], first['io_accounting_training'][key])
                 for r in reports for key in ('ssd_primary_bytes', 'ssd_completed_bytes', 'ssd_useful_bytes')]
    ranges = {key: max(r[key] for r in routes) - min(r[key] for r in routes)
              for key in ('cpu', 'gpu_ssd')}
    passed = (timing_delta <= TIME_TOLERANCE and max(spreads.values()) <= TIME_TOLERANCE
              and max(io_deltas) <= PHYSICAL_IO_TOLERANCE)
    return dict(passed=passed, paired_sampling_passed=True, logical_requests_passed=True,
        order=['real', 'shared', 'shared', 'real'], epochs_per_worker=1, updates_per_worker=1179,
        training_seconds=[r['epochs'][0]['train_seconds'] for r in reports],
        timings_order_excluded=times, medians_order_excluded=medians,
        relative_median_difference=timing_delta, within_mode_spreads=spreads,
        max_physical_io_relative_difference=max(io_deltas), time_tolerance=TIME_TOLERANCE,
        physical_io_tolerance=PHYSICAL_IO_TOLERANCE, cache_serving_routes=routes,
        cache_route_range_rows=ranges, cache_route_range_fraction_of_requests=max(ranges.values()) / routes[0]['total'],
        cache_route_exact_equality=all(r['cpu'] == routes[0]['cpu'] for r in routes),
        grid_baseline='first shared run, fixed before measurement; no selection of faster repeat',
        scope='Engineering calibration at g2/r20; feature contents and SSD base extent change. No universal equivalence or accuracy claim.')


def collect():
    execution = verify()
    reports, smokes, evidence = [], [], {}
    for mode, name in PILOTS:
        path = SOURCE / name
        status = read(path / 'status.json')
        protocol = SOURCE / 'protocols' / ('real_g2_r20.json' if mode == 'real' else 'g2_r20.json')
        require(status['complete'] and status['passed'] and status['stage'] == 'complete', 'Unaccepted pilot')
        require(status['candidate_sha256'] == NATIVE_SHA, 'Wrong worker candidate')
        require([(w['mode'], w['arm'], w['returncode']) for w in status['workers']] ==
                [('smoke', 'digit_full', 0), ('full', 'digit_full', 0)], 'Bad pilot exits')
        for run_mode, target in [('full', reports), ('smoke', smokes)]:
            report_path = path / (run_mode + '_digit_full_accepted.json')
            report = read(report_path)
            summary = read(path / (run_mode + '_summary.json'))
            require(summary['passed'] and report['passed'] and summary['smoke'] == report['smoke'] == (run_mode == 'smoke'),
                    'Unaccepted/wrong pilot extent')
            require(summary['report_sha256']['digit_full'] == sha(report_path), 'Report changed')
            require(summary['candidate_sha256'] == report['candidate_sha256'] == NATIVE_SHA, 'Wrong lineage')
            require(summary['protocol_sha256'] == report['protocol_sha256'] == sha(protocol), 'Wrong protocol')
            require(summary['feature_mode'] == report['feature_mode'] == mode, 'Wrong feature mode')
            from candidates.pa_sage_layout_shared_resume_v3.validation import no_evaluation_check
            no_evaluation_check(report, path / run_mode / 'digit_full')
            target.append(report)
            evidence[str(report_path)] = sha(report_path)
    result = evaluate(reports, smokes)
    result.update(review_candidate_sha256=execution, native_candidate_sha256=NATIVE_SHA,
        accepted_reports=evidence, original_controller_status_sha256=sha(SOURCE / 'status.json'),
        original_gate_passed=False, review_kind='post_measurement_counter_semantics_correction',
        correction='Remove exact equality of CPU/GPU serving partition; require exact logical totals and per-window reconciliation. Original 10% time and 5% physical I/O thresholds unchanged.',
        mechanism='CPU-hot rows use GPU when full_try_pin_resident reserves a complete, nonbusy page; otherwise CPU fallback. Atomic cache state can change the serving route between identical repeats.',
        mechanism_source='candidates/io_accounting_v1/native/gids_module/gids_kernel.cu:14-25,248-300')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'calibration_review.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve earlier review')
    result = collect()
    write(args.output, result)
    require(result['passed'], 'Calibration exceeds unchanged time/I/O limits')
    print('Corrected calibration review passed; no native work launched.', flush=True)


if __name__ == '__main__':
    main()
