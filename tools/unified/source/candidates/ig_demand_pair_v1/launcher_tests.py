import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from submission.v2.common import output_path
from .common import ROOT
from .launcher import run

class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(dir=ROOT/'results')
        self.addCleanup(self.folder.cleanup)
        self.output=Path(self.folder.name)/'run'
        self.calls=[];self.locks_held=False;self.admits=0

    @contextmanager
    def ready(self):
        self.locks_held=True
        try:yield None
        finally:self.locks_held=False

    def experiment(self,path):
        self.assertTrue(self.locks_held)
        self.assertEqual(path,self.output/'experiment')
        # Exercise the exact production fresh-output validator.
        path=output_path(path);path.mkdir()
        self.calls.append(path)
        return dict(stage='complete',passed=True,complete=True)

    def status(self):return json.loads((self.output/'status.json').read_text())

    def test_immediately_ready_uses_fresh_child(self):
        run(self.output,'test',self.ready,self.experiment)
        self.assertTrue(self.status()['passed']);self.assertEqual(len(self.calls),1)

    def test_wait_then_ready(self):
        @contextmanager
        def admission():
            self.admits+=1
            with self.ready():yield 'shared experiment lock busy' if self.admits==1 else None
        def sleep(seconds):
            self.assertEqual(seconds,60);self.assertFalse(self.locks_held)
            self.assertEqual(self.status()['stage'],'waiting')
            self.assertFalse((self.output/'experiment').exists())
        run(self.output,'test',admission,self.experiment,wait=True,sleep=sleep)
        self.assertEqual(self.admits,2);self.assertTrue(self.status()['complete'])
        self.assertNotIn('reason',self.status())

    def test_existing_run_is_never_overwritten(self):
        self.output.mkdir();(self.output/'status.json').write_text('original evidence')
        with self.assertRaisesRegex(RuntimeError,'Output already exists'):
            run(self.output,'test',self.ready,self.experiment)
        self.assertEqual((self.output/'status.json').read_text(),'original evidence')

    def test_early_experiment_failure_is_saved(self):
        def fail(path):raise RuntimeError('input check failed')
        with self.assertRaisesRegex(RuntimeError,'input check failed'):
            run(self.output,'test',self.ready,fail)
        self.assertEqual(self.status()['stage'],'failed');self.assertFalse(self.locks_held)

    def test_wait_signal_is_saved(self):
        @contextmanager
        def busy():yield 'busy'
        def stop(seconds):raise InterruptedError('signal 15')
        with self.assertRaises(InterruptedError):
            run(self.output,'test',busy,self.experiment,wait=True,sleep=stop)
        self.assertEqual(self.status()['stage'],'failed');self.assertFalse(self.calls)

if __name__=='__main__':unittest.main()
