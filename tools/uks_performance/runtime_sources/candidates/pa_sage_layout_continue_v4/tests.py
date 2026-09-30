"""CPU regressions using saved native reports; never open a GPU or raw device."""
import argparse
import copy
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from .common import ROOT, SOURCE, OUT, NATIVE, PILOTS, Path, read, write
from .compare import evaluate


class Tests(unittest.TestCase):
    def setUp(self):
        self.reports = [read(SOURCE / name / 'full_digit_full_accepted.json') for _, name in PILOTS]
        self.smokes = [read(SOURCE / name / 'smoke_digit_full_accepted.json') for _, name in PILOTS]

    def test_completed_abba_has_equal_requests_despite_route_variation(self):
        result = evaluate(self.reports, self.smokes)
        self.assertTrue(result['passed'])
        self.assertFalse(result['cache_route_exact_equality'])
        self.assertEqual({r['total'] for r in result['cache_serving_routes']}, {291925962})
        self.assertEqual(result['cache_route_range_rows'], dict(cpu=16, gpu_ssd=16))
        self.assertEqual(result['time_tolerance'], .10)
        self.assertEqual(result['physical_io_tolerance'], .05)

    def test_balanced_route_change_keeps_logical_trace(self):
        r = copy.deepcopy(self.reports)
        epoch = r[1]['epochs'][0]
        for interval in (epoch['training'], epoch['windows'][0]):
            interval['feature']['cpu'] += 3
            interval['feature']['gpu_ssd'] -= 3
            interval['gpu']['requests'] -= 3
            interval['useful_io']['gpu_feature_rows'] -= 3
        self.assertTrue(evaluate(r, self.smokes)['passed'])

    def test_rejects_missing_requests_and_broken_internal_accounting(self):
        for kind in ('total', 'window', 'gpu', 'useful', 'bytes', 'negative'):
            with self.subTest(kind=kind):
                r = copy.deepcopy(self.reports)
                e = r[1]['epochs'][0]
                if kind == 'total':
                    e['training']['feature']['cpu'] -= 1
                elif kind == 'window':
                    e['windows'][0]['feature']['cpu'] -= 1
                elif kind == 'gpu':
                    e['training']['gpu']['requests'] += 1
                elif kind == 'useful':
                    e['training']['useful_io']['gpu_feature_rows'] += 1
                elif kind == 'bytes':
                    r[1]['io_accounting_training']['logical_feature_bytes'] -= 512
                else:
                    e['training']['feature']['cpu'] = -1
                with self.assertRaises(RuntimeError):
                    evaluate(r, self.smokes)

    def test_rejects_changed_sampling_and_incomplete_epochs(self):
        for kind in ('shape', 'root', 'window', 'updates', 'smoke_trace', 'cache_config'):
            with self.subTest(kind=kind):
                r, s = copy.deepcopy(self.reports), copy.deepcopy(self.smokes)
                e = r[1]['epochs'][0]
                if kind == 'shape':
                    e['shapes'][0]['input_nodes'] += 1
                elif kind == 'root':
                    e['root_sha256'] = 'changed'
                elif kind == 'window':
                    e['windows'][0]['actual_edges'] += 1
                elif kind == 'updates':
                    e['updates'] -= 1
                elif kind == 'smoke_trace':
                    s[1]['audits'][0]['storage_rows'] = 'changed'
                else:
                    r[1]['cpu_cache_rows'] -= 1
                with self.assertRaises(RuntimeError):
                    evaluate(r, s)

    def test_unchanged_time_and_io_gates_still_reject_bad_results(self):
        for kind in ('time', 'repeat', 'io'):
            with self.subTest(kind=kind):
                r = copy.deepcopy(self.reports)
                if kind == 'time':
                    for i in (1, 2):
                        r[i]['epochs'][0]['train_seconds'] *= 1.2
                elif kind == 'repeat':
                    r[1]['epochs'][0]['train_seconds'] *= 1.15
                else:
                    r[1]['io_accounting_training']['ssd_primary_bytes'] *= 1.06
                self.assertFalse(evaluate(r, self.smokes)['passed'])

    def test_remaining_points_use_frozen_workers_and_reduced_checks(self):
        from .queue import cells, commands
        points = cells()
        self.assertEqual(len(points), 14)
        self.assertNotIn('g2_r20', [k for k, _ in points])
        for key, protocol in points:
            steps = commands(key, protocol)
            self.assertEqual([cmd[cmd.index('-m') + 1] for _, cmd in steps],
                             [NATIVE + '.' + m for m in ('build', 'queue', 'cli', 'cli')])
            self.assertEqual(sum('--execute' in cmd for _, cmd in steps), 1)
            self.assertNotIn('--smoke', steps[-1][1])
            self.assertEqual(read(protocol)['epochs'], 1)

    def test_controller_executes_remaining_points_and_reuses_first_shared(self):
        from . import queue
        points = queue.cells()
        row_fields = ['achieved_replica_fraction', 'mean_training_epoch_seconds', 'order_excluded_mean_seconds',
                      'speedup_vs_fresh_g2_r20', 'order_excluded_speedup_vs_g2_r20', 'layout_address_span_bytes',
                      'groups', 'padding_rows', 'group_sample_fraction']
        result = dict(passed=True, points=[dict(point=i['point'], **{k: 1 for k in row_fields})
                                          for i in read(queue.INDEX)['points']])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write(out / 'calibration_review.json', dict(passed=True))
            def run(cmd, **kwargs):
                if '--budget-output' in cmd:
                    write(Path(cmd[cmd.index('--budget-output') + 1]), dict(host_required_bytes=1, required_bytes=1))
                return subprocess.CompletedProcess(cmd, 0)
            with patch.object(queue, 'OUT', out), patch.object(queue, 'TMP', out / 'tmp'), \
                 patch.object(queue, 'verify', return_value='controller'), \
                 patch.object(queue, 'prerequisites', return_value=dict(passed=True)), \
                 patch.object(queue, 'available'), patch.object(queue.os, 'geteuid', return_value=0), \
                 patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2'), patch('sys.argv', ['queue']), \
                 patch.object(queue.subprocess, 'run', side_effect=run) as calls, \
                 patch('candidates.pa_sage_layout_shared_resume_v3.aggregate.collect', return_value=result):
                queue.main()
            status = read(out / 'status.json')
            self.assertTrue(status['passed'] and status['complete'])
            self.assertEqual(status['completed'], ['g2_r20'] + [k for k, _ in points])
            self.assertEqual(calls.call_count, 56)
            self.assertEqual((out / 'native/g2_r20').resolve(), (SOURCE / 'native/g2_r20').resolve())
            self.assertTrue((out / 'summary.csv').is_file())

    def test_controller_preserves_child_failure(self):
        from . import queue
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(queue, 'OUT', out), patch.object(queue, 'TMP', out / 'tmp'), \
                 patch.object(queue, 'verify', return_value='controller'), \
                 patch.object(queue, 'prerequisites', return_value=dict(passed=True)), \
                 patch.object(queue, 'available'), patch.object(queue.os, 'geteuid', return_value=0), \
                 patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2'), patch('sys.argv', ['queue']), \
                 patch.object(queue.subprocess, 'run', side_effect=subprocess.CalledProcessError(7, ['fixture'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    queue.main()
            status = read(out / 'status.json')
            self.assertEqual(status['stage'], 'failed')
            self.assertFalse(status['passed'] or status['complete'])
            self.assertEqual(status['completed'], ['g2_r20'])
            self.assertEqual(status['steps'][0]['returncode'], 7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(args.output, dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), native_execution=False,
        raw_ssd_access=False, method='saved-report mutation regressions and simulated controller lifecycle'))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
