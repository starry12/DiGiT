"""Sequential fresh-process native smoke; publish minimal evidence only on success."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from ae.pa_sage.common import ROOT, read, sha, write, require
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.monitor_validation import monitor_evidence
from ae.pa_sage.native_gate import current_binding, validate_reports, check_native, evidence_directory


def run(data, output):
    from ae.common import check_device, verify_release
    require(__debug__ and os.geteuid() == 0, 'Native smoke needs root and Python assertions enabled')
    data, output = Path(data).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(schema='digit-pa-native-smoke-v1', passed=False, complete=False,
                  stage='preflight', pid=os.getpid(), started_unix=time.time(), workers=[],
                  scope='real PA data, two epochs with two training and validation batches each per arm',
                  full_training_started=False, raw_writes=False)
    def save():
        status['updated_unix'] = time.time()
        write(output/'status.json', status)
    save()
    child = None
    monitor = ExternalMonitor(output/'external_gpu')
    try:
        release = verify_release()
        check_device()
        binding = current_binding(data)
        require(binding['execution_sha256'] == release, 'Release changed before smoke')
        status['binding'] = binding
        target = evidence_directory(data)
        require(not target.exists(), 'This release already has native evidence; use its receipt')
        reports, codes = {}, {}
        status['external_monitor']=monitor.start();save()
        for arm in ('gids', 'digit_full'):
            check_device()
            path = output/arm/'report.json'
            command = [sys.executable, '-u', str(ROOT/'ae/run_pa_sage.py'), 'train',
                       '--data', str(data), '--arm', arm, '--seed', '0', '--smoke', '--output', str(path)]
            worker = dict(arm=arm, command=command, started_unix=time.time(), status='running')
            status['workers'].append(worker)
            status['stage'] = arm
            with (output/(arm+'.log')).open('x') as logfile:
                child = subprocess.Popen(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
                worker['pid'] = child.pid
                save()
                code = child.wait()
                child = None
            codes[arm] = code
            worker.update(returncode=code, finished_unix=time.time(), ended_unix=time.time(), status='complete' if code == 0 else 'failed')
            save()
            require(code == 0, arm+' smoke failed; inspect '+str(output/(arm+'.log')))
            reports[arm] = read(path)
        check_device()
        require(verify_release() == release and current_binding(data) == binding, 'Source or SSD binding changed during smoke')
        status['external_monitor_returncode']=monitor.stop();save()
        require(status['external_monitor_returncode']==0,'External monitor failed')
        for arm,report in reports.items():
            report['external_monitor']=monitor_evidence(output,status,arm,read(output/arm/'resources.json'))
            write(output/arm/'report.json',report)
        validate_reports(reports, binding)
        # Copy essential evidence out of disposable run logs. The training gate
        # depends only on these small data receipts and the current release.
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.pending-', dir=target.parent))
        hashes = {}
        for arm in ('gids', 'digit_full'):
            shutil.copyfile(output/arm/'report.json', staging/(arm+'.json'))
            hashes[arm] = sha(staging/(arm+'.json'))
        receipt = dict(schema='digit-pa-native-ready-v1', passed=True, binding=binding,
                       returncodes=codes, reports_sha256=hashes, accepted_unix=time.time(),
                       full_training_completed=False, performance_result=False)
        write(staging/'ready.json', receipt)
        staging.rename(target)
        require(check_native(data)['passed'], 'Published native gate did not validate')
        status.update(passed=True, complete=True, stage='complete', native_receipt=str(target/'ready.json'),
                      native_receipt_sha256=sha(target/'ready.json'), ended_unix=time.time())
        write(output/'summary.json', status)
        save()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        monitor.stop()
        status.update(stage='failed', error=type(exc).__name__+': '+str(exc), ended_unix=time.time())
        save()
        raise
