"""Own and reap workers, validate their ready report, and require normal exit.

There is no background GPU monitor. A report explicitly records this limitation.
"""
import math
import os
import subprocess
import time
from .common import ROOT,read,sha,write,require
from candidates.pa_sage_cache_policy_v3.monitor import stop_child

POLICY=dict(enabled=False,backend='disabled',continuous_samples=0,acceptance_required=False)
STAGES=('setup','graph_ready','cache_allocated','cpu_preloaded','model_initialized','training_complete')


def observations(value,pid,started,finished):
    require(value['mode']=='cuda_checkpoints_only' and value['worker_pid']==pid and
            value['monitor_samples']==0 and not value['background_monitor_in_worker'],
            'Invalid CUDA checkpoint provenance')
    points=value['checkpoints']
    require([v['stage'] for v in points]==list(STAGES),'Missing CUDA stage observations')
    previous=started
    for point in points:
        stamp=point['time_unix']
        require(math.isfinite(stamp) and previous<=stamp<=finished,'Invalid checkpoint time')
        previous=stamp
        require(type(point['device_used_bytes']) is int and
                0<=point['device_used_bytes']<=point['device_total_bytes'], 'Invalid device observation')
    require(value['observed_peak_device_used_bytes']==max(p['device_used_bytes'] for p in points),
            'Checkpoint maximum differs from observations')
    return value


def validate_evidence(report):
    require(report['monitor']==POLICY,'Continuous-monitoring policy differs')
    evidence=report['process_supervision']
    require(evidence['normal_exit'] and evidence['worker_returncode']==report['worker_returncode']==0 and
            evidence['worker_pid']>0 and evidence['controller_pid']>0 and
            evidence['worker_pid']!=evidence['controller_pid'],'Worker did not exit normally')
    for key in ('report_sha256','resources_sha256'):
        require(isinstance(evidence[key],str) and len(evidence[key])==64,'Missing process evidence hash')
    observations(report['resource_observations'],evidence['worker_pid'],
                 evidence['started_unix'],evidence['finished_unix'])
    return report


def run_worker(command,attempt,arm,notify=lambda **kw:None):
    folder=attempt/'worker';child=None
    row=dict(arm=arm,command=command,status='starting',controller_pid=os.getpid())
    try:
        with (attempt/'worker.log').open('x') as log:
            row['started_unix']=time.time()
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row.update(pid=child.pid,status='running');notify(**row)
            released=False;deadline=None
            while child.poll() is None:
                ready=folder/'worker_ready.json'
                if ready.exists() and not released:
                    value=read(ready);digest=sha(folder/'report.json')
                    require(value['passed'] and value['report_sha256']==digest,'Worker ready report changed')
                    raw=read(folder/'report.json');resources=read(folder/'resources.json')
                    require(raw['passed'] and raw['resource_observations']==resources,'Worker resource report differs')
                    observations(resources,child.pid,row['started_unix'],time.time())
                    resource_sha=sha(folder/'resources.json')
                    write(folder/'release_worker.json',dict(passed=True,report_sha256=digest,
                        continuous_monitoring=False))
                    released=True;deadline=time.monotonic()+180
                if deadline is not None:require(time.monotonic()<deadline,'Worker resource release timed out')
                time.sleep(.1)
            row.update(returncode=child.wait(),finished_unix=time.time(),status='exited');notify(**row)
        require(row['returncode']==0 and released,'Worker failed or exited without ready/release handshake')
        require(sha(folder/'report.json')==digest and sha(folder/'resources.json')==resource_sha,
                'Worker evidence changed after release')
        raw=read(folder/'report.json')
        raw.update(worker_returncode=0,monitor=dict(POLICY),process_supervision=dict(
            normal_exit=True,worker_returncode=0,worker_pid=child.pid,controller_pid=os.getpid(),
            started_unix=row['started_unix'],finished_unix=row['finished_unix'],
            report_sha256=digest,resources_sha256=resource_sha))
        validate_evidence(raw);write(attempt/'supervised.json',raw)
        return raw,row
    finally:stop_child(child)
