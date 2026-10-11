"""Independent full-epoch affinity control; reuse the frozen native backend."""
import math
import os
from pathlib import Path
import statistics
from candidates.pa_sage_legacy_align_v1 import common as legacy
from candidates.pa_sage_legacy_align_v1.common import (
    ROOT, PY, BINARY, GPU_UUID, sha, read, write, require, check_inputs, setup,
    progress, hashlib, json, time,
)

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/pa_sage_cpu_affinity_20260926_v1'
MODULE = 'candidates.pa_sage_cpu_affinity_v1'


def verify():
    manifest = read(HERE / 'manifest.json')
    require(legacy.verify() == manifest['parent_manifest_sha256'], 'Frozen parent changed')
    for rel, digest in manifest['files'].items():
        require(sha(ROOT / rel) == digest, 'Candidate changed: ' + rel)
    require(sha(OUT / 'inputs.json') == manifest['inputs_sha256'], 'Input binding changed')
    require(sha(OUT / 'cpu_checks.json') == manifest['cpu_checks_sha256'], 'CPU checks changed')
    return sha(HERE / 'manifest.json')


def validate_completion(report):
    legacy.validate_completion(report)
    require(report['protocol'] == read(HERE / 'protocol.json'), 'Report protocol differs from candidate')
    require(report['block_index_bits'] == 32, 'Expected legacy int32 model blocks')
    affinity = report['affinity']
    require(affinity['policy'] in ('unbound', 'one_core'), 'Unknown CPU policy')
    if affinity['policy'] == 'one_core':
        for stage in ('at_training_start', 'at_training_end'):
            require(bool(affinity[stage]) and all(v == [affinity['cpu']] for v in affinity[stage].values()),
                    'Training thread escaped CPU affinity')
    else:
        require(all(v == affinity['before'][k] for k, v in affinity['at_training_start'].items()
                    if k in affinity['before']), 'Unbound control affinity was changed')
    for name in ('block_conversion_host_seconds', 'model_process_cpu_seconds', 'epoch_process_cpu_seconds'):
        require(math.isfinite(report['timing'][name]) and report['timing'][name] >= 0, 'Invalid ' + name)
    require(bool(report['cpu_frequency_samples']), 'Missing CPU observations')
    require(all(r['main_frequency_khz'] > 0 for r in report['cpu_frequency_samples']), 'Invalid CPU frequency')
    return True


def paired_summary(reports):
    p = read(HERE / 'protocol.json')
    require([r['affinity']['policy'] for r in reports] == p['full_order'], 'Wrong full-run order')
    require(all(not r['smoke'] for r in reports), 'Smoke cannot become full performance result')
    for report in reports:
        validate_completion(report)
    for field in ('root_order_sha256', 'candidate_sha256', 'input_binding_sha256', 'binary_sha256'):
        require(len(set(r[field] for r in reports)) == 1, 'Unmatched full runs: ' + field)
    require(all(r['protocol'] == reports[0]['protocol'] for r in reports), 'Different protocols')
    groups = {}
    for policy in ('unbound', 'one_core'):
        selected = [r for r in reports if r['affinity']['policy'] == policy]
        timing = {k: statistics.mean(r['timing'][k] for r in selected)
                  for k, v in selected[0]['timing'].items() if k.endswith('_seconds') and isinstance(v, (float, int))}
        groups[policy] = dict(runs=len(selected), mean_seconds=timing,
            e2e_seconds=[r['timing']['e2e_seconds'] for r in selected],
            logical_feature_requests=[r['logical_feature_requests'] for r in selected],
            ssd_read_bytes=[r['device_io']['completed_bytes'] for r in selected],
            gpu_cache_hit_rate=[r['gpu_cache']['hit_rate'] for r in selected],
            main_frequency_mean_ghz=[statistics.mean(s['main_frequency_khz'] for s in r['cpu_frequency_samples']) / 1e6
                                     for r in selected])
    before = groups['unbound']['mean_seconds']; after = groups['one_core']['mean_seconds']
    effects = {k: dict(unbound=before[k], one_core=after[k], delta=after[k]-before[k],
                      reduction_percent=100*(1-after[k]/before[k]) if before[k] else None)
               for k in before}
    return dict(passed=True, complete=True, groups=groups, effects_seconds=effects,
                e2e_speedup=before['e2e_seconds']/after['e2e_seconds'],
                order=p['full_order'], epochs_per_run=1, full_updates_per_run=1179,
                candidate_sha256=reports[0]['candidate_sha256'],
                limits=['Two process repetitions per policy at seed 0, not multi-seed accuracy evidence',
                        'Same roots and configuration; GPU neighbor sampling can vary across processes',
                        'Sampling is nested in inclusive fetch; host spans are not independent CUDA kernels',
                        'CPU affinity and observed frequency covary; frequency is periodically sampled, not an exact average',
                        'Cache/model recreated in each worker; OS page cache is not flushed'])
