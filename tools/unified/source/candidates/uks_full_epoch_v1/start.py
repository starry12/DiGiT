"""Serial, guarded UKS full-epoch GIDS/DiGiT comparison."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path('/home/embed/digit')
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from candidates.uks_full_epoch_v1.common import OUT,HERE,PY,require,write,verify,inputs,schedule,aggregate

def blockers():
    from candidates.cache_epoch_v1.start import blockers as prior
    result=prior()
    units=subprocess.check_output(['systemctl','list-units','digit-followup-cache-epoch*','digit-hot-ab*','--all','--no-pager','--plain','--no-legend'],text=True,timeout=30)
    for line in units.splitlines():
        fields=line.split()
        if len(fields)>=4 and fields[2] in ('active','activating','deactivating') and fields[3]!='exited':result.append(fields[0])
    return result

def control(folder):
    require(os.geteuid()==0,'Root required');source=verify();inputs();folder.mkdir(parents=True,exist_ok=False)
    with open('/tmp/digit-full-epoch-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(passed=False,complete=False,source_sha256=source,output=str(folder),started_unix=time.time(),completed=[])
        def save(**kw):
            state.update(kw,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',state)
            print(json.dumps(state),flush=True)
        def stop(sig,frame):raise InterruptedError('Controller interrupted: '+str(sig))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        child=None;reports={}
        try:
            for name in schedule():
                while True:
                    active=blockers()
                    if not active:break
                    save(stage='waiting',next_stage=name,blockers=active,worker_pid=None);time.sleep(60)
                require(verify()==source,'Source changed')
                cmd=[PY,'-B','-u','-m','candidates.uks_full_epoch_v1.worker','--arm',name,'--output',str(folder/name)]
                write(folder/(name+'_command.json'),dict(command=cmd))
                with (folder/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    save(stage=name,worker_pid=child.pid,blockers=[],log=str(folder/(name+'.log')))
                    while child.poll() is None:
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:save(stage=name,worker_pid=child.pid)
                    code=child.returncode;child=None
                write(folder/(name+'_exit.json'),dict(returncode=code,finished_unix=time.time()))
                require(code==0,'Worker failed: '+name)
                reports[name]=json.loads((folder/name/'report.json').read_text())
                state['completed'].append(name);save(stage=name+'_complete',worker_pid=None)
            summary=aggregate(reports);summary['source_sha256']=source
            write(folder/'summary.json',summary);write(OUT/'summary.json',summary)
            save(stage='complete',passed=True,complete=True,finished_unix=time.time())
        except BaseException as exc:
            save(stage='failed',error=str(exc),traceback=traceback.format_exc(),finished_unix=time.time())
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:child.kill();child.wait()

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--control',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.control:
        require(args.output is not None,'Missing output');return control(args.output)
    source=verify();inputs();checks=json.loads((OUT/'cpu_checks.json').read_text())
    require(checks['passed'] and checks['source_sha256']==source,'Missing current CPU checks')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid());folder=OUT/'runs'/stamp
    unit='digit-full-epoch-'+stamp
    cmd=['/usr/bin/systemd-run','--unit='+unit,'--service-type=exec','--property=WorkingDirectory='+str(ROOT),
         '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=120',
         '--property=LimitMEMLOCK=infinity','--property=MemoryMax=384G','--property=MemorySwapMax=0','--property=UMask=0022',
         '--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
         '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PY,'-B','-u',str(HERE/'start.py'),'--control','--output',str(folder)]
    plan=dict(command=cmd,unit=unit+'.service',output=str(folder),source_sha256=source,stages=schedule(),execute=args.execute)
    print(json.dumps(plan,indent=2),flush=True)
    if args.execute:
        require(os.geteuid()==0,'Run with sudo');subprocess.run(cmd,check=True);write(OUT/'launch.json',plan)

if __name__=='__main__':main()
