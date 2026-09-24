"""CPU fault injection for the unchanged author nvidia-smi monitor."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'ae/pa_sage/gpu_monitor.py').is_file())

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'ae/pa_sage' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

gpu = module('gpu_monitor')
control = module('monitor_control')

class MonitorTests(unittest.TestCase):
    def test_query_timeout_units_and_device(self):
        with patch.object(gpu.subprocess, 'check_output', return_value='12, 7\n') as query:
            sample = gpu.sample(2)
        query.assert_called_once_with(['nvidia-smi', '-i', '2', '--query-gpu=memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True, timeout=5)
        self.assertEqual(sample['device_used_bytes'], 12 * 2**20)
        self.assertEqual(sample['utilization_percent'], 7)

    def test_bad_samples_rejected(self):
        for text in ('-1,0', '0,101', '0,-1', '1,0,0', 'N/A,0', '1,0\n2,0'):
            with self.subTest(text=text), patch.object(gpu.subprocess, 'check_output', return_value=text):
                with self.assertRaises(ValueError): gpu.sample(2)

    def exercise(self, samples, first_error=False):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'monitor'
            with patch.object(gpu.signal, 'signal'), patch.object(gpu.threading, 'Event') as event:
                event.return_value.is_set.side_effect = [False] * len(samples) + [True]
                with patch.object(gpu, 'sample', side_effect=samples):
                    if first_error:
                        with self.assertRaises(subprocess.TimeoutExpired): gpu.main(out, 2, .5)
                    else: gpu.main(out, 2, .5)
            summary = json.loads((out / 'summary.json').read_text())
            records = [json.loads(line) for line in (out / 'samples.jsonl').read_text().splitlines()]
            return summary, records, (out / 'ready.json').exists()

    def test_first_query_timeout_blocks_readiness(self):
        s, records, ready = self.exercise([subprocess.TimeoutExpired('nvidia-smi', 5)], True)
        self.assertFalse(ready)
        self.assertFalse(s['passed'])
        self.assertTrue(s['complete'])
        self.assertEqual(s['samples'], 0)
        self.assertEqual(len(records), 1)

    def test_later_timeout_retained_and_sampling_continues(self):
        good = dict(time_unix=1., query_started_unix=.99, device_used_bytes=1024)
        s, records, ready = self.exercise([good, subprocess.TimeoutExpired('nvidia-smi', 5), good])
        self.assertTrue(ready)
        self.assertEqual(s['samples'], 2)
        self.assertEqual(len(records), 3)
        self.assertEqual(len(s['errors']), 1)
        self.assertIn('TimeoutExpired', s['errors'][0]['error'])
        self.assertFalse(s['passed'])

    def test_successful_completion(self):
        s, records, ready = self.exercise([dict(time_unix=1., query_started_unix=.99, device_used_bytes=1024)])
        self.assertTrue(ready and s['passed'] and s['complete'])
        self.assertEqual(s['errors'], [])

    def test_wrong_ready_pid_stops_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'monitor'; out.mkdir()
            (out / 'ready.json').write_text(json.dumps(dict(pid=10, passed=True)))
            child = Mock(pid=11); child.poll.return_value = None; child.wait.return_value = 0
            with patch.object(control.subprocess, 'Popen', return_value=child):
                monitor = control.ExternalMonitor(out)
                with self.assertRaisesRegex(RuntimeError, 'Bad external monitor receipt'): monitor.start()
                child.terminate.assert_called_once()
                self.assertIsNone(monitor.log)

    def test_stop_escalates_only_after_termination_timeout(self):
        monitor = control.ExternalMonitor('/unused')
        child = Mock(); child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('monitor', 10), -9]
        monitor.process = child
        self.assertEqual(monitor.stop(), -9)
        child.terminate.assert_called_once(); child.kill.assert_called_once()

if __name__ == '__main__': unittest.main()
