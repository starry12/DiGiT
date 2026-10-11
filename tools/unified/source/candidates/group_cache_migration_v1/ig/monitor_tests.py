"""CPU-only regression for slow queries and unchanged evidence rejection."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from candidates.ig_monitor_v1.review_tests import fixture
from .monitor_review import assess_monitor, POLICY
from .monitor_control import ExternalMonitor


def slow_fixture(delay=12.0):
    value = fixture()
    for i, sample in enumerate(value['records']):
        start = 10.0 + i * 2.0 + (delay if i > 6 else 0)
        end = start + (delay if i == 6 else .01)
        sample.update(query_started_unix=start, query_finished_unix=end, time_unix=end)
    last = value['records'][-1]['time_unix']
    value['summary'].update(query_timeout_seconds=30., period_after_query_seconds=2., finished_unix=last+.1)
    value['worker']['finished_unix'] = last+.5
    value['resources']['checkpoints'][-1]['time_unix'] = last-.1
    value['ready']['first_sample'] = copy.deepcopy(value['records'][0])
    return value


class MonitorRevision(unittest.TestCase):
    def test_slow_query_accepted_with_reported_policy(self):
        result = assess_monitor(**slow_fixture())
        self.assertTrue(result['strict_monitor_passed'])
        self.assertEqual(result['review_policy'], POLICY)
        self.assertEqual(result['error_count'], 0)

    def test_over_deadline_and_gaps_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'query timing'):
            assess_monitor(**slow_fixture(31))
        value = slow_fixture()
        for sample in value['records'][8:]:
            for field in ('time_unix', 'query_started_unix', 'query_finished_unix'):
                sample[field] += 45
        value['summary']['finished_unix'] += 45
        value['worker']['finished_unix'] += 45
        value['resources']['checkpoints'][-1]['time_unix'] += 45
        with self.assertRaisesRegex(RuntimeError, 'monitoring gap'):
            assess_monitor(**value)

    def test_errors_identity_and_memory_still_reject(self):
        value = slow_fixture()
        error = dict(time_unix=9.5, error='query failed', error_kind='nvml_query_error')
        value['records'].insert(0, error)
        value['summary'].update(errors=[error], passed=False)
        with self.assertRaisesRegex(RuntimeError, 'zero errors'):
            assess_monitor(**value)
        value = slow_fixture()
        value['records'][5]['physical_gpu_uuid'] = 'GPU-other'
        with self.assertRaisesRegex(RuntimeError, 'sample process/device identity'):
            assess_monitor(**value)
        value = slow_fixture()
        value['summary'].update(peak_rss_bytes=70*2**20, sampler_peak_rss_bytes=60*2**20)
        with self.assertRaisesRegex(RuntimeError, 'RSS exceeds'):
            assess_monitor(**value)

    def test_missing_endpoint_and_abnormal_exit_reject(self):
        value = slow_fixture()
        value['worker']['finished_unix'] += 40
        value['resources']['checkpoints'][-1]['time_unix'] += 40
        with self.assertRaisesRegex(RuntimeError, 'cover training lifetime'):
            assess_monitor(**value)
        value = slow_fixture()
        value['returncode'] = 2
        with self.assertRaisesRegex(RuntimeError, 'finish normally'):
            assess_monitor(**value)

    def test_controller_passes_new_limits_to_existing_sampler(self):
        with tempfile.TemporaryDirectory() as td:
            child = MagicMock()
            child.poll.return_value = 2
            child.wait.return_value = 2
            with patch('subprocess.Popen', return_value=child) as launch:
                with self.assertRaisesRegex(RuntimeError, 'before readiness'):
                    ExternalMonitor(Path(td)/'monitor').start()
            command = launch.call_args[0][0]
            self.assertEqual(command[command.index('--period')+1], '2')
            self.assertEqual(command[command.index('--query-timeout')+1], '30')
            self.assertIn('/home/embed/digit/candidates/ig_monitor_v1/gpu_monitor.py', command)
