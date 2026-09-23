"""Controller-owned monitor lifecycle; never called by a training worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class ExternalMonitor:
    def __init__(self,output):
        self.output=Path(output);self.process=None;self.log=None
    def start(self):
        self.log=self.output.with_suffix('.log').open('x')
        script=Path(__file__).resolve().with_name('gpu_monitor.py')
        try:
            self.process=subprocess.Popen([sys.executable,'-u',str(script),'--output',str(self.output),
                '--gpu',os.environ.get('CUDA_VISIBLE_DEVICES','2')],stdout=self.log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('External GPU monitor exited before readiness')
                ready=self.output/'ready.json'
                if ready.exists():
                    value=json.loads(ready.read_text())
                    if value['pid']!=self.process.pid or not value['passed']:raise RuntimeError('Bad external monitor receipt')
                    return value
                time.sleep(.1)
            raise RuntimeError('External GPU monitor readiness timed out')
        except BaseException:
            self.stop();raise
    def stop(self):
        if self.process is None:
            if self.log is not None:self.log.close();self.log=None
            return None
        if self.process.poll() is None:self.process.terminate()
        try:code=self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:self.process.kill();code=self.process.wait()
        if self.log is not None:self.log.close();self.log=None
        return code
