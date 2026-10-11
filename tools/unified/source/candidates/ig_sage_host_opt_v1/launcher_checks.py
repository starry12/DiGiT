"""CPU-only lifecycle checks. No native controller, GPU, or systemd is run."""
import builtins
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('ig_rerun_start', Path(__file__).with_name('start.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Checks(unittest.TestCase):
    def test_original_entry_and_fresh_result(self):
        cmd = m.experiment_command(Path('/tmp/example'))
        self.assertIn('candidates.ig_sage_host_opt_v1.controller', cmd)
        self.assertEqual(cmd[5:], ['--mode', 'representative', '--model', 'sage',
                                  '--gpu', '2', '--output', '/tmp/example/experiment'])
        with self.assertRaises(RuntimeError):
            m.validate_folder(m.ROOT / 'results/ig_sage_window_20260922_v1')

    def test_admission_rejects_busy_wrong_gpu_and_low_host(self):
        for first, second in [('wrong, 13', ''), (m.GPU_UUID + ', 2048', ''),
                              (m.GPU_UUID + ', 13', '1234')]:
            with mock.patch.object(m, 'host', return_value=440 * 2**30), \
                 mock.patch.object(m.subprocess, 'check_output', side_effect=[first, second]):
                with self.assertRaises(RuntimeError):
                    m.admission()
        with mock.patch.object(m, 'host', return_value=319 * 2**30), \
             mock.patch.object(m.subprocess, 'check_output') as query:
            with self.assertRaises(RuntimeError):
                m.admission()
            query.assert_not_called()

    def test_worker_failure_writes_nonpassing_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            folder = out / 'native/failure'
            folder.mkdir(parents=True)
            m.write(folder / 'identity.json', {'fixture': True})
            child = mock.Mock(pid=1234)
            child.wait.return_value = 7
            child.poll.return_value = 7
            with mock.patch.object(m, 'OUT', out), \
                 mock.patch.object(m.os, 'geteuid', return_value=0), \
                 mock.patch.dict(m.os.environ, {'TMUX': 'mock-session'}), \
                 mock.patch.object(m, 'identities', return_value={'fixture': True}), \
                 mock.patch.object(m.subprocess, 'Popen', return_value=child):
                self.assertEqual(m.worker(folder), 1)
            receipt = m.read(folder / 'worker_exit.json')
            self.assertFalse(receipt['passed'])
            self.assertEqual(receipt['controller_returncode'], 7)
            self.assertFalse((folder / 'completion_review.json').exists())

    def test_supervisor_requires_successful_closed_worker(self):
        for result in ('passed', 'failed', 'missing'):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as td:
                out = Path(td)
                folder = out / 'native/fixture'
                original_open = builtins.open
                def local_open(path, *args, **kwargs):
                    if str(path) == '/tmp/digit-pa-sage-512b-pair-controller.lock':
                        path = out / 'fixture.lock'
                    return original_open(path, *args, **kwargs)
                def fake_run(cmd, **kwargs):
                    if 'new-session' in cmd and result != 'missing':
                        ok = result == 'passed'
                        m.write(folder / 'worker_exit.json', dict(
                            passed=ok, complete=ok, returncode=0 if ok else 1))
                    return subprocess.CompletedProcess(cmd, 1 if 'has-session' in cmd else 0)
                with mock.patch.object(m, 'OUT', out), \
                     mock.patch.object(m.os, 'geteuid', return_value=0), \
                     mock.patch.object(m, 'identities', return_value={'fixture': True}), \
                     mock.patch.object(m, 'admission', return_value={'fixture': True}), \
                     mock.patch('builtins.open', side_effect=local_open), \
                     mock.patch.object(m.subprocess, 'run', side_effect=fake_run):
                    self.assertEqual(m.supervise(folder), 0 if result == 'passed' else 1)
                state = m.read(folder / 'status.json')
                self.assertEqual(state['passed'], result == 'passed')
                self.assertEqual(state['complete'], result == 'passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
