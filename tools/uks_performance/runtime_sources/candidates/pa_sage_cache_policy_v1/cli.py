"""Plan and bounded CPU validation only; current grid is never stopped or edited."""
import argparse
import os
import resource
import time
from .common import ROOT, HERE, OUT, Path, read, write_new, sha, verify, require


def protected_snapshot():
    folders = ['pa_sage_layout_continue_v4', 'pa_sage_layout_shared_resume_v3',
               'pa_sage_bidir_native_v2', 'io_accounting_v1', 'pa_sage_layout_sweep_v1']
    paths = []
    for name in folders:
        folder = ROOT / 'candidates' / name
        paths += list(folder.glob('*.py'))
        if (folder / 'manifest.json').exists():
            paths.append(folder / 'manifest.json')
    paths.append(ROOT / 'ae/papers/runtime/models.py')
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def budget(p):
    manifest = read(ROOT / p['layout']['base'] / 'final/bundle/manifest.json')
    rows, nodes = manifest['feature']['num_storage_rows'], p['graph']['nodes']
    return dict(kind='component planning only, not live native admission',
        cpu_feature_bytes_per_arm={k: a['cpu_feature_bytes'] for k, a in p['arms'].items()},
        gpu_feature_cache_bytes_per_arm={k: a['gpu_feature_cache_bytes'] for k, a in p['arms'].items()},
        exact_cpu_row_map_gpu_bytes=rows * 4, logical_slot_build_host_bytes=nodes * 4,
        score_vector_bytes_each=nodes * 8, topk_working_arrays_note='partition/sort and masks need additional temporary host memory',
        exact_cpu_row_map_note='common mapping metadata, not feature cache; native admission must include this overhead for every arm',
        cpu_host_floor_bytes=192 * 2**30, gpu_native_admission_passed=False)


def plan(output):
    from .protocol import compile_protocol
    p = compile_protocol()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'protocol.json', p)
    write_new(output / 'budget_components.json', budget(p))
    archive = ROOT / 'data/pa_sage_cache_v1/shared/profile_provenance.json'
    old = read(archive)
    write_new(output / 'input_audit.json', dict(
        graph= p['graph'], layout_manifest_sha256=p['layout']['manifest_sha256'],
        real_pool_receipt_sha256=p['features']['receipt_sha256'],
        original_frequency_source=old['frequency_source'],
        original_revpr_provenance=str(archive),
        original_scores_accepted_for_this_comparison=False,
        reason='The archived frequency profile uses bootstrap layout and measured seed/order; RevPR graph provenance is not bound to this fixed bidirectional training graph. Prepare a separate new profile and graph-bound ranks.',
        large_arrays_read=False, raw_ssd_access=False, preparation_started=False))
    print('Protocol and component budgets written; no large preparation or native execution.', flush=True)


def cpu_check(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU checks require CUDA_VISIBLE_DEVICES=')
    require(os.environ.get('OPENBLAS_NUM_THREADS') == '1' and os.environ.get('OMP_NUM_THREADS') == '1', 'Use one CPU thread')
    state_path = ROOT / 'results/pa_sage_layout_continue_20260925_v4/status.json'
    queue_before = read(state_path)
    require(not queue_before['stage'].startswith(('native_', 'budget_')),
            'Grid is entering/running a native measurement; defer even lightweight CPU tests')
    os.nice(19)
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {max(allowed)})
    # Independent Python process, one core and at most 120 seconds CPU time.
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    output.mkdir(parents=True, exist_ok=False)
    before = protected_snapshot()
    write_new(output / 'protected_before.json', before)
    started = time.time()
    from .tests import run
    checks = run()
    from .fixture import exercise
    fixture = exercise(output / 'fixture')
    after = protected_snapshot()
    require(after == before, 'Active experiment sources changed')
    import torch
    require(not torch.cuda.is_initialized(), 'CPU checks initialized CUDA')
    write_new(output / 'protected_after.json', after)
    write_new(output / 'summary.json', dict(passed=True, regression=checks, fixture=fixture,
        protected_sources_unchanged=True, native_execution=False, raw_ssd_access=False,
        cuda_initialized=False, cpu_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.nice(0),
        wall_seconds=time.time() - started, peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        active_grid_before={k: queue_before.get(k) for k in ('pid', 'stage', 'completed')},
        active_grid_after={k: read(state_path).get(k) for k in ('pid', 'stage', 'completed')}))


def baseline(output):
    from .metrics import from_native_report
    paths = ['calibration/real_first', 'calibration/real_second']
    folder = ROOT / 'results/pa_sage_layout_shared_resume_20260925_v3'
    records = []
    for name in paths:
        base = folder / name
        report_path = base / 'full_digit_full_accepted.json'
        summary, status = read(base / 'full_summary.json'), read(base / 'status.json')
        require(summary['passed'] and status['passed'] and status['complete'] and
                all(w['returncode'] == 0 for w in status['workers']) and
                summary['report_sha256']['digit_full'] == sha(report_path), 'Changed/unaccepted baseline')
        records.append(dict(path=str(report_path), report_sha256=sha(report_path), **from_native_report(read(report_path))))
    write_new(output, dict(kind='existing native evidence, metrics extraction only',
        cache_policy_comparison_complete=False, new_native_workers=0, records=records,
        reason='Old hot profile and current backend; these runs validate denominator extraction but are not the new four-arm comparison.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'cpu-check', 'baseline', 'native-preview'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify()
    if args.action == 'plan':
        plan(args.output or OUT / 'plan')
    elif args.action == 'cpu-check':
        cpu_check(args.output or OUT / 'cpu_checks')
    elif args.action == 'baseline':
        baseline(args.output or OUT / 'existing_baseline_metrics.json')
    else:
        from .protocol import compile_protocol
        import json
        print(json.dumps(compile_protocol()['implementation'], indent=2))
        print('No --execute route in this CPU preparation version; native backend work remains.')


if __name__ == '__main__':
    main()
