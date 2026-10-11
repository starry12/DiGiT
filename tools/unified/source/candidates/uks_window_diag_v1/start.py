"""One cold-cache pair, followed by separate stage diagnostics if no speedup."""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from candidates.uks_window_diag_v1.common import HERE,OUT,PY,LOCKS,require,write,verify,inputs,aggregate
from candidates.uks_window_diag_v1.monitor import snapshot

def blockers():
    from candidates.uks_full_epoch_v1.start import blockers as previous
    busy=previous()
    # Preserve the already-running serial chain, including gaps between its
    # workers, instead of making the migration's next worker lose its lock.
    r=subprocess.run(['systemctl','list-units','digit-large-resume*','digit-ig-after-cl*',
                      '--all','--no-pager','--plain','--no-legend'],capture_output=True,text=True,check=True,timeout=20)
    for line in r.stdout.splitlines():
        x=line.split()
        if len(x)>=4 and x[2] in ('active','activating','deactivating') and x[3]!='exited':busy.append(x[0])
    return busy

def run_jobs(run,save_summary):
    reports={}
    for arm in ('gids','digit'):reports[arm]=run(arm,arm,'off')
    summary=aggregate(reports);save_summary(summary)
    if summary['diagnostics_required']:
        for arm in ('gids','digit'):reports['stages_'+arm]=run('stages_'+arm,arm,'stages')
        summary=aggregate(reports);save_summary(summary)
    return summary

def control(folder,pause_ukl=False):
    source=verify();inputs();folder.mkdir(parents=True,exist_ok=False)
    state=dict(passed=False,complete=False,source_sha256=source,output=str(folder),started_unix=time.time(),completed=[])
    child=None
    lease=folder/"ukl_pause.json"
    def save(**fields):
        state.update(fields,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',state)
    def stopped(sig,frame):raise InterruptedError('Controller signal '+str(sig))
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    evidence=[]
    def observe(stage):
        value=snapshot();value['stage']=stage
        with (folder/'host_observations.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        evidence.append(value)
    def save_summary(value):
        oom=[v['vm'].get('oom_kill',0) for v in evidence]
        value.update(source_sha256=source,host_oom_kill_delta=max(oom)-min(oom) if oom else 0,
                     host_oom_observed=(max(oom)>min(oom)) if oom else False,
                     clean_environment_proven=False,
                     host_monitor_scope='Read-only host snapshots every 5 seconds and each training window; no absence-of-interference claim')
        write(folder/'summary.json',value);write(OUT/'summary.json',value)
    try:
        with open('/tmp/digit-uks-window-diag-controller.lock','a+') as unique:
            fcntl.flock(unique,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if pause_ukl:
                from candidates.uks_window_diag_v1.pause import begin,validate
                save(stage='pausing_ukl')
                begin(lease)
                stack=ExitStack();files=[]
            while not pause_ukl:
                busy=blockers()
                if busy:
                    save(stage='waiting',blockers=busy,poll_seconds=60);time.sleep(60);continue
                stack=ExitStack()
                try:
                    files=[]
                    for path in LOCKS:
                        require(Path(path).is_file() and not Path(path).is_symlink(),'Unsafe lock path')
                        f=stack.enter_context(open(path,'rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);files.append(f)
                except BlockingIOError:
                    stack.close();save(stage='waiting',blockers=['shared experiment lock busy'],poll_seconds=60);time.sleep(60);continue
                except BaseException:stack.close();raise
                break
            # Hold the same open-file-description locks through all workers.
            # They inherit the descriptors rather than acquire conflicting locks.
            with stack:
                fds=tuple(f.fileno() for f in files)
                def run(name,arm,mode):
                    nonlocal child
                    require(verify()==source,'Source changed')
                    cmd=[PY,'-B','-u','-m','candidates.uks_window_diag_v1.worker','--arm',arm,
                         '--profile',mode,'--output',str(folder/name)]
                    if pause_ukl:
                        validate(lease);cmd+=['--pause-lease',str(lease)]
                    else:cmd+=['--lock-fds',','.join(map(str,fds))]
                    write(folder/(name+'_command.json'),dict(command=cmd))
                    observe(name+'_before')
                    with (folder/(name+'.log')).open('x') as log:
                        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=fds)
                        save(stage=name,worker_pid=child.pid,blockers=[])
                        while child.poll() is None:
                            if pause_ukl:validate(lease)
                            try:child.wait(timeout=5)
                            except subprocess.TimeoutExpired:observe(name);save(stage=name,worker_pid=child.pid)
                        code=child.returncode;child=None
                    observe(name+'_after')
                    write(folder/(name+'_exit.json'),dict(returncode=code,finished_unix=time.time()))
                    require(code==0,'Worker failed: '+name)
                    report=json.loads((folder/name/'report.json').read_text())
                    state['completed'].append(name);save(stage=name+'_complete',worker_pid=None)
                    return report
                summary=run_jobs(run,save_summary)
                if pause_ukl:validate(lease)
                require(not summary['diagnostics_required'] or summary['diagnostics_complete'],'Missing required diagnostics')
                save(stage='complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as exc:
        save(stage='failed',error=repr(exc),finished_unix=time.time());raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if pause_ukl:
            from candidates.uks_window_diag_v1.pause import restore
            restore(lease)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--control',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--pause-ukl',action='store_true')
    a=p.parse_args()
    if a.control:
        require(os.geteuid()==0 and __debug__,'Root/assertions required');return control(a.output,a.pause_ukl)
    source=verify();inputs()
    checks=json.loads((OUT/'cpu_checks.json').read_text());require(checks['passed'] and checks['source_sha256']==source,'Missing matching checks')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid());folder=OUT/'runs'/stamp
    unit='digit-windowdiag-uks-'+stamp
    cmd=['systemd-run','--unit='+unit,'--service-type=exec','--property=WorkingDirectory='+str(ROOT),
         '--property=Restart=no','--property=KillMode=control-group','--property=LimitMEMLOCK=infinity',
         '--property=MemoryMax=384G','--property=MemorySwapMax=0','--property=UMask=0022',
         '--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
         PY,'-B','-u',str(HERE/'start.py'),'--control','--output',str(folder)]
    if a.pause_ukl:
        cmd.insert(2,'--property=ExecStopPost='+PY+' -B '+str(HERE/'pause.py')+' --restore '+str(folder/'ukl_pause.json'))
        cmd.insert(2,'--property=TimeoutStopSec=120')
        cmd+=['--pause-ukl']
    plan=dict(pause_ukl=a.pause_ukl,source_sha256=source,unit=unit,output=str(folder),command=cmd,
              schedule=('freeze UKL validation, UKS pair and optional stage diagnostics, automatically thaw UKL' if a.pause_ukl else 'current UKL/CL/IG chain, then UKS pair and optional stage diagnostics'),execute=a.execute)
    if a.execute:
        require(os.geteuid()==0,'Run with sudo');subprocess.run(cmd,check=True);write(OUT/'launch.json',plan)
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
