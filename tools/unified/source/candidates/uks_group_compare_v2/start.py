"""Queue UKS matched BFS GIDS/DiGiT comparisons; five paired rounds, median reporting."""
import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/home/embed/digit');sys.path.insert(0,str(ROOT))
from candidates.uks_group_compare_v2.common import *
from candidates.uks_group_compare_v2.report import validate_worker,summarize,coverage_summary
from candidates.uks_group_compare_v2.resources import target_gpu_busy
from candidates.pa_sage_mixed_group_v1.queue import active_ae
HERE=OUT
MODULE='candidates.uks_group_compare_v2'
STAGES=stages()


def wait_available(save,need_gpu=True):
    while True:
        busy=active_ae()
        apps=target_gpu_busy() if need_gpu else []
        if apps:busy.append('Target GPU2 compute PIDs: '+','.join(apps))
        if not busy:
            try:
                with exclusive():pass
                return
            except RuntimeError as e:
                if 'Another experiment is active' not in str(e):raise
                busy.append(str(e))
        save(stage='waiting_for_resources',reason=busy)
        time.sleep(60)


def control(folder):
    require(os.geteuid()==0,'Native workers require local sudo')
    source=verify();v1.check_prepared(v1.DEFAULT_OUTPUT);script=sha(__file__)
    with open('/tmp/digit-uks-group-compare-v2-controller.lock','a+') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder.mkdir(parents=True,exist_ok=False);(OUT/'runs').mkdir(exist_ok=True)
        state=dict(passed=False,complete=False,controller_pid=os.getpid(),started_unix=time.time(),
                   output=str(folder),source_sha256=source,controller_sha256=script,completed=[])
        def save(**kw):
            state.pop('reason',None);state.update(kw,updated_unix=time.time())
            write(folder/'status.json',state);write(HERE/'latest.json',state);print(json.dumps(state),flush=True)
        def stop(signum,frame):raise RuntimeError('Controller interrupted by signal '+str(signum))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        child=None;reports={};evidence=[]
        try:
            # Preparation has no CUDA/raw SSD access. Native stages independently
            # wait for GPU2; unrelated GPU1/3 tasks do not block either stage.
            while not (DEFAULT_OUTPUT/'prepared.json').exists():
                require(not DEFAULT_OUTPUT.exists(),'Partial v2 output exists; preserve it and inspect before resuming')
                wait_available(save,need_gpu=False)
                require(verify()==source,'Source changed before preparation')
                log=folder/('prepare_'+str(time.time_ns())+'.log')
                cmd=[PY,'-B','-u','-m','candidates.uks_group_reuse_v2.prepare','--execute']
                write(folder/'prepare_command.json',cmd)
                with log.open('x') as stream:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    save(stage='preparation',worker_pid=child.pid,log=str(log))
                    while True:
                        try:code=child.wait(timeout=30);break
                        except subprocess.TimeoutExpired:
                            progress_path=v2.OUT/'preparation_status.json'
                            progress=read(progress_path) if progress_path.exists() else None
                            save(stage='preparation',preparation=progress,
                                 builder_progress=preparation_tail())
                    child=None
                write(folder/'prepare_exit.json',dict(returncode=code,finished_unix=time.time()))
                if code and 'Another experiment is active' in log.read_text() and not DEFAULT_OUTPUT.exists():
                    time.sleep(60);continue
                require(code==0,'Preparation failed; inspect '+str(log))
            receipts=check_prepared();prepared=prepared_hashes()
            coverage=coverage_summary(receipts)
            write(folder/'coverage.json',coverage);write(HERE/'coverage.json',coverage)
            state['completed'].append('preparation');save(stage='preparation_accepted',
                  prepared_sha256=prepared,coverage=coverage,worker_pid=None)
            for stage,mode,variant,number in STAGES:
                attempt=0
                while True:
                    wait_available(save);attempt+=1
                    require(verify()==source and sha(__file__)==script,'Source/controller changed')
                    require(prepared_hashes()==prepared,'Prepared receipt changed')
                    dest=OUT/'runs'/(folder.name+'_'+stage+'_'+str(attempt))
                    if mode is None:
                        cmd=[PY,'-B','-u','-m',MODULE+'.gpu_check']
                    else:
                        cmd=[PY,'-B','-u','-m',MODULE+'.worker','--variant',variant,'--mode',mode,'--round',str(number or 0),'--output',str(dest),'--execute']
                    log=folder/(stage+'_'+str(attempt)+'.log');start=time.time()
                    write(folder/(stage+'_'+str(attempt)+'_command.json'),dict(command=cmd,started_unix=start))
                    with log.open('x') as stream:
                        child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                        save(stage=stage,worker_pid=child.pid,worker_output=str(dest),log=str(log),worker_progress=None,worker_elapsed_seconds=0)
                        while True:
                            try:code=child.wait(timeout=30);break
                            except subprocess.TimeoutExpired:
                                progress=read(dest/'progress.json') if (dest/'progress.json').exists() else None
                                save(stage=stage,worker_elapsed_seconds=time.time()-start,worker_progress=progress)
                        child=None
                    write(folder/(stage+'_'+str(attempt)+'_exit.json'),dict(returncode=code,finished_unix=time.time()))
                    # Only retry a resource race before any native output exists.
                    text=log.read_text()
                    race=('Another experiment is active' in text or 'Target GPU2 is occupied' in text)
                    if code and race and not dest.exists():
                        save(stage='waiting_for_resources',reason='Resource acquired by another job before worker admission')
                        time.sleep(60);continue
                    require(code==0,'Worker failed: '+stage+'; see '+str(log))
                    if mode is None:
                        receipt=read(OUT/'gpu_checks.json')
                        require(receipt['passed'] and receipt['source_sha256']==source,'Missing GPU acceptance')
                        write(folder/'accepted_gpu_check.json',receipt)
                    else:
                        r=read(dest/'report.json');validate_worker(r,mode,variant,source,prepared,number)
                        reports[stage]=r;evidence.append(dict(stage=stage,report=str(dest/'report.json'),sha256=sha(dest/'report.json')))
                    state['completed'].append(stage);save(stage=stage+'_accepted',worker_pid=None)
                    break
            result=summarize(reports);result['evidence']=evidence;result['coverage']=coverage
            require(verify()==source and sha(__file__)==script,'Source changed before summary')
            check_prepared();require(prepared_hashes()==prepared,'Overlay changed')
            write(folder/'summary.json',result);write(HERE/'summary.json',result)
            save(stage='complete',passed=True,complete=True,worker_pid=None,finished_unix=time.time())
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
            save(stage='failed',error=repr(error),worker_pid=None);raise


def preparation_tail():
    result={}
    for mode in ('build','validate'):
        path=v2.OUT/(DEFAULT_OUTPUT.name+'_'+mode+'.log')
        if path.exists():
            with path.open('rb') as f:
                f.seek(0,2);f.seek(max(0,f.tell()-4096));lines=f.read().decode(errors='replace').splitlines()
            result[mode]=lines[-1] if lines else 'initializing'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--control',action='store_true')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.control:
        require(args.output is not None,'Missing controller output');control(args.output);return
    source=verify()
    from candidates.uks_group_reuse_v2.prepare import plan as preparation_plan
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    unit='digit-uks-group-compare-v2-'+stamp;folder=HERE/'runs'/stamp
    command=['/usr/bin/systemd-run','--unit='+unit,'--service-type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=Restart=no','--property=KillMode=control-group',
        '--property=TimeoutStopSec=60','--property=LimitMEMLOCK=infinity','--property=UMask=0022',
        '--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),'--setenv=OMP_NUM_THREADS=16',
        '--setenv=MKL_NUM_THREADS=16','--setenv=OPENBLAS_NUM_THREADS=1',
        PY,'-B','-u',str(Path(__file__).resolve()),'--control','--output',str(folder)]
    plan=dict(unit=unit+'.service',output=str(folder),command=command,stages=['preparation']+STAGES,
              source_sha256=source,preparation=preparation_plan(DEFAULT_OUTPUT),
              poll_seconds=60,raw_ssd_writes=False,execute=args.execute)
    print(json.dumps(plan,indent=2),flush=True)
    if args.execute:
        require(os.geteuid()==0,'Run with local sudo')
        subprocess.run(command,check=True);write(HERE/'launch.json',plan)

if __name__=='__main__':main()
