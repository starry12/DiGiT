import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/ig_after_cl_v1'
PY='/home/embed/miniconda3/envs/gids/bin/python'
IG_SHA='1505dc4744a73d8959fba337bf2b8699a593469231c226eec3d7b91927733a4c'
RESUME_SHA='0763e75241836bea2d6289a4f9baf6e4b8c1017f6af9dfcd3d9811f687b1b728'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise RuntimeError(message)

def dependency_reason(spec,outer,migration,service,summaries):
    # Never infer completion from an idle GPU or a gap between workers.
    require(outer.get('source_sha256')==spec['source_sha256'] and
            outer.get('output')==spec['output'] and outer.get('datasets')==['UKL','CL'],
            'Bound UKL/CL run identity changed')
    active=service.get('ActiveState')
    if active in ('active','activating','deactivating','reloading'):
        return 'waiting for bound UKL/CL service to finish; stage='+str(migration.get('stage',outer.get('stage')))
    if service.get('LoadState')!='not-found':
        if active!='inactive' or service.get('Result')!='success' or service.get('ExecMainStatus')!='0':
            return 'UKL/CL service unsuccessful or unknown; IG remains blocked'
    if not (outer.get('passed') is True and outer.get('complete') is True and
            migration.get('passed') is True and migration.get('complete') is True):
        return 'UKL/CL completion not accepted; IG remains blocked'
    for ds in ('UKL','CL'):
        record=migration.get('datasets',{}).get(ds,{})
        if not (record.get('prepared') is True and record.get('native_complete') is True and
                summaries.get(ds,{}).get('passed') is True):
            return ds+' final comparison not accepted; IG remains blocked'
    return None

def current_reason(spec):
    folder=Path(spec['output'])
    outer=read(folder/'status.json')
    migration=read(folder/'migration/status.json') if (folder/'migration/status.json').exists() else {}
    result=subprocess.run(['systemctl','show',spec['unit'],'--property=LoadState,ActiveState,Result,ExecMainStatus'],
                          text=True,capture_output=True,timeout=20,check=True)
    service=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    summaries={ds:read(folder/'migration'/(ds+'_summary.json'))
               for ds in ('UKL','CL') if (folder/'migration'/(ds+'_summary.json')).exists()}
    return dependency_reason(spec,outer,migration,service,summaries)

def verify():
    from candidates.ig_demand_pair_v1.common import verify as ig
    from candidates.group_cache_resume_v1.start import verify as resume
    require(ig()==IG_SHA and resume()==RESUME_SHA,'Frozen experiment or dependency source changed')
    manifest=read(HERE/'manifest.json')
    for name,digest in manifest['files'].items():require(sha(HERE/name)==digest,'Queue source changed: '+name)
    return sha(HERE/'manifest.json')

@contextmanager
def admission(spec,check=current_reason,resource=None):
    reason=check(spec)
    if reason is not None:
        yield reason
        return
    if resource is None:
        from candidates.ig_demand_pair_v1.launcher import admission as resource
    with resource() as busy:
        # Recheck while holding shared locks before releasing the experiment.
        yield busy if busy is not None else check(spec)

def control(output,plan_path):
    from candidates.ig_demand_pair_v1.launcher import run
    from candidates.ig_demand_pair_v1.controller import execute
    from candidates.ig_demand_pair_v1.gpu_check import require_accepted
    from candidates.ig_demand_pair_v1.common import environment
    require(os.geteuid()==0 and __debug__,'Use sudo with assertions enabled')
    source=verify();plan=read(plan_path)
    require(plan['queue_sha256']==source and plan['output']==str(output),'Wrong queue plan')
    require_accepted();os.environ.update(environment(2))
    def stopped(signum,frame):raise InterruptedError('Queue signaled '+str(signum))
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    with open('/tmp/digit-ig-after-cl-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(output,source,lambda:admission(plan['dependency']),
            lambda child:execute('representative',child,2,'sage'),wait=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--control',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path)
    a=p.parse_args()
    if a.control:return control(a.output,a.plan)
    from ae.common import write
    from candidates.ig_demand_pair_v1.gpu_check import require_accepted
    source=verify();require_accepted()
    previous=read(ROOT/'results/group_cache_resume_v1/latest.json')
    require(previous['datasets']==['UKL','CL'] and previous['source_sha256']==RESUME_SHA,'Wrong predecessor')
    dep={key:previous[key] for key in ('unit','output','source_sha256')}
    reason=current_reason(dep)
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    folder=OUT/'runs'/stamp;unit='digit-ig-after-cl-'+stamp;plan_path=OUT/'plans'/(stamp+'.json')
    command=[PY,'-B','-u',str(HERE/'start.py'),'--control','--output',str(folder),'--plan',str(plan_path)]
    plan=dict(queue_sha256=source,ig_sha256=IG_SHA,dependency=dep,unit=unit,output=str(folder),
              command=command,poll_seconds=60,initial_wait_reason=reason,
              schedule='bound UKL then CL accepted and service exited, then frozen IG 20+300 pair')
    if a.execute:
        require(os.geteuid()==0,'Starting the background service requires sudo')
        # A preview/failed sudo call must not leave a misleading latest pointer.
        write(plan_path,plan)
        subprocess.run(['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
                        '--property=WorkingDirectory='+str(ROOT),'--property=KillMode=control-group']+command,check=True)
        write(OUT/'latest.json',plan)
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
