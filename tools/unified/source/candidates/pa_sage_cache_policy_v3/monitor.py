"""Own only this child's process group and the existing nvidia-smi monitor."""
import os
import signal
import subprocess
import sys
import time
from .common import ROOT,HERE,GPU,read,sha,write,require
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.monitor_validation import monitor_evidence


class SupervisedMonitor(ExternalMonitor):
    def start(self):
        self.log=self.output.with_suffix('.log').open('x')
        try:
            self.process=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'monitor_worker.py'),
                '--output',str(self.output),'--gpu',GPU,'--parent',str(os.getpid())],
                stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                require(self.process.poll() is None,'GPU monitor exited before readiness')
                if (self.output/'ready.json').exists():
                    value=read(self.output/'ready.json')
                    require(value['passed'] and value['pid']==self.process.pid,'Wrong monitor readiness receipt')
                    return value
                time.sleep(.1)
            raise RuntimeError('GPU monitor readiness timed out')
        except BaseException:self.stop();raise


def stop_child(child):
    if child is None or child.poll() is not None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def run_monitored(command,attempt,arm,notify=lambda **kw:None,monitor_factory=SupervisedMonitor):
    mon_dir=attempt/'monitor';mon_dir.mkdir()
    worker_dir=attempt/'worker';monitor=monitor_factory(mon_dir/'external_gpu');child=None
    row=dict(arm=arm,command=command,status='starting')
    try:
        ready=monitor.start()
        with (attempt/'worker.log').open('x') as log:
            row['started_unix']=time.time()
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row.update(pid=child.pid,status='running');notify(**row)
            released=False;deadline=None
            while child.poll() is None:
                receipt=worker_dir/'worker_ready.json'
                if receipt.exists() and not released:
                    evidence=read(receipt)
                    require(evidence['passed'] and sha(worker_dir/'report.json')==evidence['report_sha256'],'Worker report changed before monitoring closed')
                    code=monitor.stop()
                    summary=read(mon_dir/'external_gpu/summary.json')
                    require(code==0 and summary['passed'] and summary['complete'] and not summary['errors'],'nvidia-smi monitoring failed')
                    write(worker_dir/'release_worker.json',dict(passed=True,monitor_returncode=code))
                    released=True;deadline=time.monotonic()+180
                if not released:
                    require(monitor.process.poll() is None,'Monitor exited while worker was active')
                if deadline is not None:require(time.monotonic()<deadline,'Worker did not release native resources')
                time.sleep(.25)
            row.update(returncode=child.wait(),finished_unix=time.time(),status='exited');notify(**row)
        require(row['returncode']==0 and released,'Worker failed or exited before monitor handshake')
        raw=read(worker_dir/'report.json')
        require(sha(worker_dir/'report.json')==read(worker_dir/'worker_ready.json')['report_sha256'],'Worker report changed after release')
        status=dict(pid=os.getpid(),workers=[row],external_monitor=ready,external_monitor_returncode=code)
        evidence=monitor_evidence(mon_dir,status,arm,read(worker_dir/'resources.json'))
        raw.update(worker_returncode=row['returncode'],monitor=dict(evidence,passed=True,backend='nvidia-smi',
            query_timeout_seconds=5,max_gap_enforced=False,summary_sha256=sha(mon_dir/'external_gpu/summary.json')))
        write(attempt/'monitored.json',raw)
        return raw,row
    finally:
        # Never kill an unrelated user process or stop the original grid unit.
        stop_child(child)
        monitor.stop()
