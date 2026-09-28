"""Exercise supervision, lifetime, ownership and failure policy without a GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


class MonitorTests(unittest.TestCase):
    def test_buffered_response_after_deadline_is_rejected(self):
        import threading
        from candidates.ig_monitor_v1.gpu_monitor import read_response, QueryFailure
        with self.assertRaises(QueryFailure) as caught:
            read_response(None, None, threading.Event(), time.monotonic()-1, b'{}\n')
        self.assertEqual(caught.exception.kind, 'query_timeout')

    def test_nvml_v2_memory_used_excludes_reserved(self):
        import ctypes
        from candidates.ig_monitor_v1.gpu_monitor import NvmlBackend
        backend = object.__new__(NvmlBackend)
        backend.gpu = '2'
        backend.uuid = 'GPU-fake-memory-api'
        backend.driver_version = backend.nvml_version = 'fake'
        backend.handle = ctypes.c_void_p()
        def fill_memory(handle, pointer):
            value = pointer._obj
            self.assertEqual(value.version, ctypes.sizeof(NvmlBackend.Memory) | (2 << 24))
            value.total, value.reserved, value.free, value.used = 1000, 100, 880, 20
        def fill_utilization(handle, pointer):
            pointer._obj.gpu = 3
        backend.get_memory, backend.get_utilization = fill_memory, fill_utilization
        sample = backend.sample()
        self.assertEqual(sample['device_used_bytes'], 20)
        self.assertEqual(sample['device_reserved_bytes'], 100)
        self.assertEqual(sample['device_free_bytes'], 880)
        self.assertEqual(sample['device_total_bytes'], 1000)
        self.assertEqual(sample['memory_api'], 'nvmlDeviceGetMemoryInfo_v2')

    def run_case(self, scenario, normal_stop=False):
        with tempfile.TemporaryDirectory(prefix='digit-monitor-test-') as tmp:
            output = Path(tmp)/'monitor'
            started = time.monotonic()
            process = subprocess.Popen([sys.executable, '-u', str(HERE/'tests.py'), '--child', scenario, str(output)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                if normal_stop:
                    deadline = time.monotonic()+5
                    while not (output/'ready.json').exists():
                        self.assertIsNone(process.poll())
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(.01)
                    time.sleep(.15 if scenario != 'exit_after_response' else .02)
                    process.terminate()
                stdout, stderr = process.communicate(timeout=7)
                self.assertEqual(stdout, '', stdout)
                self.assertEqual(stderr, '', stderr)
                summary = json.loads((output/'summary.json').read_text())
                ready = json.loads((output/'ready.json').read_text())
                records = [json.loads(s) for s in (output/'samples.jsonl').read_text().splitlines()]
                self.assertLess(time.monotonic()-started, 6)
                self.assertTrue(summary['complete'])
                self.assertTrue(summary['sampler_stopped'])
                self.assertFalse(Path('/proc/%d' % summary['sampler_pid']).exists())
                self.assertEqual(summary['pid'], process.pid)
                self.assertEqual(summary['parent_pid'], os.getpid())
                self.assertEqual(summary['peak_rss_bytes'], summary['supervisor_peak_rss_bytes']+summary['sampler_peak_rss_bytes'])
                self.assertLessEqual(summary['peak_rss_bytes'], 64*2**20)
                self.assertEqual(summary['errors'], [r for r in records if 'error' in r])
                self.assertEqual(summary['samples'], len([r for r in records if 'error' not in r]))
                for record in records:
                    self.assertEqual(record['monitor_pid'], process.pid)
                    self.assertEqual(record['monitor_parent_pid'], os.getpid())
                    self.assertEqual(record['sampler_pid'], summary['sampler_pid'])
                return process.returncode, summary, ready, records
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_clean_lifetime_and_identity(self):
        code, summary, ready, records = self.run_case('normal', True)
        self.assertEqual(code, 0)
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['errors'], [])
        self.assertGreaterEqual(summary['samples'], 2)
        self.assertTrue(ready['passed'])
        self.assertEqual(ready['first_sample'], records[0])
        self.assertEqual(summary['backend'], 'test_fake_backend')
        for i, record in enumerate(records, 1):
            self.assertEqual(record['sequence'], i)
            self.assertEqual(record['sampler_parent_pid'], summary['pid'])
            self.assertEqual(record['physical_gpu_uuid'], summary['physical_gpu_uuid'])
            self.assertEqual(record['monitor_rss_bytes'], record['supervisor_rss_bytes']+record['sampler_rss_bytes'])
            self.assertLessEqual(record['query_started_unix'], record['query_finished_unix'])
            self.assertLessEqual(record['query_finished_unix'], record['time_unix'])

    def assert_failed(self, scenario, kind, started=False):
        code, summary, ready, records = self.run_case(scenario)
        self.assertNotEqual(code, 0)
        self.assertFalse(summary['passed'])
        self.assertEqual(ready['passed'], started)
        self.assertEqual(len(summary['errors']), 1)
        self.assertEqual(summary['errors'][0]['error_kind'], kind)
        self.assertEqual(summary['samples'], int(started))

    def test_hung_initialization_bounded(self):
        self.assert_failed('hang', 'query_timeout')

    def test_hung_query_after_readiness_never_passes(self):
        self.assert_failed('hang_after_ready', 'query_timeout', True)

    def test_partial_response_does_not_block_readline(self):
        self.assert_failed('partial', 'query_timeout')

    def test_disconnect_before_readiness(self):
        self.assert_failed('disconnect', 'sampler_disconnected')

    def test_disconnect_after_readiness(self):
        self.assert_failed('disconnect_after_ready', 'sampler_disconnected', True)

    def test_query_error_never_passes(self):
        self.assert_failed('error', 'nvml_query_error', True)

    def test_wrong_gpu_rejected(self):
        self.assert_failed('wrong_gpu', 'identity_error')

    def test_wrong_pid_rejected(self):
        self.assert_failed('wrong_pid', 'identity_error')

    def test_wrong_parent_rejected(self):
        self.assert_failed('wrong_parent', 'identity_error')

    def test_wrong_sequence_rejected(self):
        self.assert_failed('wrong_sequence', 'identity_error')

    def test_changed_gpu_uuid_rejected(self):
        self.assert_failed('changing_uuid', 'identity_error', True)

    def test_unexpected_sampler_exit_cannot_pass_shutdown(self):
        code, summary, ready, records = self.run_case('exit_after_response', True)
        self.assertNotEqual(code, 0)
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['errors'][0]['error_kind'], 'unexpected_sampler_exit')

    def test_intentional_stop_during_query_is_bounded(self):
        code, summary, ready, records = self.run_case('hang_after_ready', True)
        self.assertEqual(code, 0)
        self.assertTrue(summary['passed'])
        self.assertTrue(summary['stop_during_query'])
        self.assertEqual(summary['samples'], 1)
        self.assertEqual(summary['errors'], [])


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--child':
        sys.path.insert(0, str(HERE))
        from gpu_monitor import run_monitor
        sys.exit(run_monitor(sys.argv[3], '2', period=(1. if sys.argv[2] == 'exit_after_response' else .03), timeout=.4,
                 sampler_command=[sys.executable, '-u', str(HERE/'fake_sampler.py'), sys.argv[2]],
                 expected_backend='test_fake_backend'))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MonitorTests)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    print(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
                          failures=len(result.failures), errors=len(result.errors)), indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
