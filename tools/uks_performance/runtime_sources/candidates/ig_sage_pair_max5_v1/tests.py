"""CPU-only selection/command/export guards; no GPU or SSD experiments."""
import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path
from .common import ROOT, read
from .controller import ARMS, jobs, command, summarize, make_row, emit


def rows():
    # Same-round maximum is 24/10=2.4, distinct from min/min=20/10
    # and the invalid cross-round maximum 45/10.
    values = {'gids_1': [10., 10., 10.], 'gids_2': [8., 8., 8.], 'gids_3': [15., 15., 15.],
              'gids_4': [6., 7., 7.], 'gids_5': [7., 7., 7.],
              'digit_1': [6., 6., 8.], 'digit_2': [3., 3., 4.], 'digit_3': [10., 10., 10.],
              'digit_4': [6., 6., 6.], 'digit_5': [4., 5., 5.]}
    result = []
    for job in jobs():
        windows = values[job['mode']]
        result.append(dict(mode=job['mode'], arm=job['arm'], repetition=job['repetition'], accepted=True,
                           measured_batches=300, warmup_batches=20, windows_seconds=windows,
                           seconds=sum(windows), ssd_completed_bytes=1024, host_stages_seconds={}))
    return result


class PairMaximumFiveTests(unittest.TestCase):
    def test_maximum_same_round_ratio(self):
        data = rows(); stats = summarize(data)
        self.assertEqual(stats['selected_round'], 2)
        self.assertEqual(stats['selected_pair']['gids_run'], 'gids_2')
        self.assertEqual(stats['selected_pair']['digit_run'], 'digit_2')
        self.assertEqual(stats['max_observed_speedup'], 2.4)
        self.assertNotEqual(stats['max_observed_speedup'], 20. / 10.)
        self.assertNotEqual(stats['max_observed_speedup'], 45. / 10.)
        self.assertEqual(stats['arms']['gids']['all_runs_seconds'], [30., 24., 45., 20., 21.])
        self.assertEqual(stats['arms']['digit_full']['all_runs_seconds'], [20., 10., 30., 18., 14.])
        self.assertFalse(stats['interference_free_proven'])

    def test_select_before_rounding(self):
        data = rows()
        for row in data:
            if row['mode'] == 'gids_4':
                row['windows_seconds'] = [18. * 2.400001 / 3] * 3
                row['seconds'] = sum(row['windows_seconds'])
        stats = summarize(data)
        self.assertEqual(stats['selected_round'], 4)
        self.assertEqual('%.2f' % stats['max_observed_speedup'], '2.40')

    def test_first_round_wins_exact_ties(self):
        data = rows()
        for row in data:
            row['windows_seconds'] = [10., 10., 10.] if row['arm'] == 'gids' else [5., 5., 10.]
            row['seconds'] = sum(row['windows_seconds'])
        self.assertEqual(summarize(data)['selected_round'], 1)

    def test_negative_result_retained(self):
        data = rows()
        for row in data:
            if row['arm'] == 'digit_full':
                row['windows_seconds'] = [x * 4 for x in row['windows_seconds']]
                row['seconds'] *= 4
        value = summarize(data)
        self.assertLess(value['max_observed_speedup'], 1.)
        self.assertLess(value['selected_pair']['training_time_reduction_percent'], 0.)

    def test_missing_extra_reordered_runs_rejected(self):
        for data in (rows()[:-1], rows() + [rows()[0]], rows()[::-1]):
            with self.assertRaises(Exception):
                summarize(data)

    def test_wrong_arm_or_duplicate_repetition_rejected(self):
        for key, value in [('arm', 'digit_full'), ('repetition', 2)]:
            data = rows(); data[0][key] = value
            with self.assertRaises(Exception):
                summarize(data)

    def test_invalid_run_not_dropped(self):
        for key, value in [('accepted', False), ('measured_batches', 299), ('warmup_batches', 0)]:
            data = rows(); data[0][key] = value
            with self.assertRaises(Exception):
                summarize(data)

    def test_nonfinite_or_synthetic_total_rejected(self):
        for value in (math.nan, math.inf, -1., 15.):
            data = rows(); data[0]['seconds'] = value
            with self.assertRaises(Exception):
                summarize(data)

    def test_worker_commands_preserve_both_native_arms(self):
        self.assertEqual([j['arm'] for j in jobs()], ['gids', 'digit_full', 'digit_full', 'gids', 'gids',
                                                      'digit_full', 'digit_full', 'gids', 'gids', 'digit_full'])
        for job in jobs():
            out = Path('/tmp/paired-max5-test-only')
            cmd = command(job, out)
            self.assertIn('candidates.ig_sage_host_telemetry_v1.worker', cmd)
            self.assertEqual(cmd[cmd.index('--arm') + 1], job['arm'])
            self.assertEqual(cmd[cmd.index('--output') + 1], str(out / job['mode'] / job['arm']))
            self.assertNotIn('/usr/bin/taskset', cmd)
            self.assertNotIn('--smoke', cmd)

    def test_export_ten_workers_and_five_rounds(self):
        data = rows()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            emit(dict(rows=data, statistics=summarize(data), limits=['CPU fixture only']), out)
            with (out / 'summary.csv').open() as f:
                values = list(csv.DictReader(f))
            self.assertEqual(len(values), 10)
            self.assertEqual({r['run'] for r in values if r['selected_pair'] == 'True'}, {'gids_2', 'digit_2'})
            with (out / 'pairs.csv').open() as f:
                pairs = list(csv.DictReader(f))
            self.assertEqual(len(pairs), 5)
            self.assertEqual([r['round'] for r in pairs if r['selected'] == 'True'], ['2'])
            text = (out / 'README.md').read_text()
            self.assertIn('五轮最大观察加速比：2.40×（第2轮）', text)
            self.assertIn('GIDS 24.00 秒（gids_2）与 DiGiT 10.00 秒（digit_2）', text)
            self.assertEqual({r['system'] for r in values}, {'GIDS', 'DiGiT'})

    def test_actual_saved_reports_support_both_row_formats_and_pair_metadata(self):
        # Only parse historical accepted JSON in CPU memory; never reuse these
        # times in the new experiment, which always launches ten fresh workers.
        old = ROOT / 'results/ig_sage_host_telemetry_20260927_v1'
        run = Path(read(old / 'launch.json')['output']) / 'experiment'
        saved = {arm: read(run / (mode + '_' + arm + '_accepted.json'))
                 for arm, mode in [('gids', 'gids_1'), ('digit_full', 'digit_1')]}
        data = [make_row(job, saved[job['arm']]) for job in jobs()]
        stats = summarize(data)
        self.assertEqual([stats['arms'][arm]['count'] for arm in ARMS], [5, 5])
        from candidates.ig_sage_host_telemetry_v1.validation import pair_check
        self.assertTrue(pair_check(saved, False)['passed'])
        bad = copy.deepcopy(saved)
        bad['digit_full']['training']['roots_sha256'] = 'wrong'
        with self.assertRaises(Exception):
            pair_check(bad, False)


if __name__ == '__main__':
    unittest.main()
