"""Controller-owned lifecycle for the independent persistent NVML monitor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class ExternalMonitor:
    def __init__(self, output):
        self.output = Path(output)
        self.process = None
        self.log = None

    def start(self):
        self.log = self.output.with_suffix('.log').open('x')
        script = Path(__file__).resolve().with_name('gpu_monitor.py')
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '2')
        try:
            self.process = subprocess.Popen([sys.executable, '-u', str(script), '--output', str(self.output),
                                             '--gpu', gpu], stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                ready = self.output/'ready.json'
                if ready.exists():
                    value = json.loads(ready.read_text())
                    if (value.get('pid') != self.process.pid or value.get('parent_pid') != os.getpid()
                            or not value.get('passed') or value.get('backend') != 'persistent_nvml_ctypes_v1'
                            or value.get('physical_gpu_index') != int(gpu)):
                        raise RuntimeError('Invalid persistent monitor readiness: ' + str(value))
                    if self.process.poll() is not None:
                        raise RuntimeError('Persistent monitor exited immediately after readiness')
                    return value
                if self.process.poll() is not None:
                    raise RuntimeError('Persistent monitor exited before readiness; inspect ' + str(self.output))
                time.sleep(.05)
            raise RuntimeError('Persistent monitor readiness timed out; inspect ' + str(self.output))
        except BaseException:
            self.stop()
            raise

    def poll(self):
        return None if self.process is None else self.process.poll()

    def stop(self):
        if self.process is None:
            if self.log is not None:
                self.log.close()
                self.log = None
            return None
        if self.process.poll() is None:
            self.process.terminate()
        try:
            code = self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait(timeout=2)
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None
        return code
