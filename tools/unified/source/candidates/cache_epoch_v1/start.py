"""Serial UKS/UKL/CL full-epoch cache study; wait for PA/reviewer queues."""
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
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from candidates.cache_epoch_v1.common import OUT, PY, verify, require, write


def blockers():
    from candidates.pa_sage_mixed_group_v1.queue import active_ae
    from candidates.uks_group_first_cpu2_v1.resources import prior_controller_busy,target_gpu_busy
    active=active_ae()+prior_controller_busy()
    units=subprocess.run(['systemctl','list-units','digit-followup-uks*','--all','--no-pager','--plain','--no-legend'],
                         capture_output=True,text=True,timeout=30,check=True)
    for line in units.stdout.splitlines():
        parts=line.split()
        if len(parts)>=4 and parts[2] in ('active','activating','deactivating') and parts[3]!='exited':active.append(parts[0])
    if target_gpu_busy():active.append('GPU2 occupied')
    return active


def summarize_uks(folder):
    reports={arm:json.loads((folder/('uks_'+arm)/'report.json').read_text()) for arm in ('gids','digit')}
    a,b=reports['gids'],reports['digit']
    require(a['passed'] and b['passed'] and a['roots_sha256']==b['roots_sha256'],'UKS pair rejected')
    for key in ('cpu_feature_bytes','gpu_feature_cache_bytes'):
        require(a['cache'][key]==b['cache'][key],'Unequal '+key)
    return dict(passed=True,speedup=a['seconds']/b['seconds'],aggregation='one native pair, no max selection',
                arms={k:dict(seconds=v['seconds'],serving=v['training']['serving'],cache=v['cache']) for k,v in reports.items()},
                profile=b['cache_epoch'],accuracy_evaluated=False,full_epoch_training=False)


def control(folder, datasets=('UKS',)):
    require(os.geteuid()==0,'Local sudo required for guarded GPU/SSD workers')
    source=verify();folder.mkdir(parents=True,exist_ok=False)
    with open('/tmp/digit-cache-epoch-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(source_sha256=source,passed=False,complete=False,started_unix=time.time(),
                   controller_pid=os.getpid(),output=str(folder),requested_datasets=list(datasets),datasets={})
        def save(**kw):
            state.update(kw,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',state)
            print(json.dumps(state),flush=True)
        child=None
        def stop(signum,frame):raise InterruptedError('Controller signal '+str(signum))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        def wait_available():
            while True:
                active=blockers()
                if not active:return
                save(stage='waiting_for_resources',blockers=active,worker_pid=None)
                time.sleep(60)
        def run(name,cmd):
            nonlocal child
            wait_available();require(verify()==source,'Source changed')
            write(folder/(name+'_command.json'),dict(command=cmd,time=time.time()))
            with (folder/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(stage=name,blockers=[],worker_pid=child.pid,log=str(folder/(name+'.log')))
                while child.poll() is None:
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        save(stage=name,worker_pid=child.pid)
                code=child.returncode;child=None
            write(folder/(name+'_exit.json'),dict(returncode=code,finished_unix=time.time()))
            require(code==0,'Worker failed: '+name)
        try:
            for ds in datasets:
                try:
                    if ds=='UKS':
                        for arm in ('gids','digit'):
                            run('uks_'+arm,[PY,'-B','-u','-m','candidates.cache_epoch_v1.uks_worker',
                                '--arm',arm,'--output',str(folder/('uks_'+arm))])
                        result=summarize_uks(folder)
                    else:
                        prior=json.loads((OUT/'UKS_native_accepted.json').read_text())
                        require(prior['passed'] and prior['source_sha256']==source,
                                'Accept this version on UKS before adapting UKL/CL')
                        run(ds.lower(),[PY,'-B','-u','-m','candidates.cache_epoch_v1.'+ds.lower()+'.start','--execute'])
                        paths=sorted((OUT/ds/'gpu').glob('cache_comparison_*.json'))
                        require(paths,'Missing native cache comparison')
                        result=json.loads(paths[-1].read_text())
                    write(folder/(ds+'_summary.json'),result)
                    if ds=='UKS':
                        write(OUT/'UKS_native_accepted.json',dict(passed=result['passed'],source_sha256=source,
                              summary=str(folder/(ds+'_summary.json'))))
                    state['datasets'][ds]=dict(passed=result['passed'],summary=str(folder/(ds+'_summary.json')))
                except (RuntimeError,ValueError,OSError) as error:
                    state['datasets'][ds]=dict(passed=False,error=str(error))
                    write(folder/(ds+'_failure.json'),dict(error=repr(error),traceback=traceback.format_exc()))
                save(stage=ds+'_finished',worker_pid=None)
            save(stage='complete',complete=True,passed=all(v['passed'] for v in state['datasets'].values()),finished_unix=time.time())
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:child.kill();child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--control',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--datasets',nargs='+',choices=('UKS','UKL','CL'),default=['UKS'])
    args=parser.parse_args()
    require(len(set(args.datasets))==len(args.datasets),'Duplicate dataset')
    if args.control:
        require(args.output is not None,'Missing output');return control(args.output,tuple(args.datasets))
    source=verify();checks=json.loads((OUT/'cpu_checks.json').read_text())
    require(checks['passed'] and checks['source_sha256']==source,'Missing current CPU checks')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    folder=OUT/'runs'/stamp;unit='digit-followup-cache-epoch-'+stamp
    cmd=['/usr/bin/systemd-run','--unit='+unit,'--service-type=exec','--property=WorkingDirectory='+str(ROOT),
         '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=120',
         '--property=LimitMEMLOCK=infinity','--property=MemoryMax=384G','--property=MemorySwapMax=0',
         '--property=UMask=0022','--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
         '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
         PY,'-B','-u',str(Path(__file__).resolve()),'--control','--output',str(folder),'--datasets',*args.datasets]
    plan=dict(command=cmd,unit=unit+'.service',output=str(folder),source_sha256=source,execute=args.execute,
              datasets=args.datasets,profile_epochs=1,profile_seed=23,cache_audit_seeds=[0,1],
              native_pairs_per_dataset=1,native_warmup_batches=20,native_measured_batches=300,
              matched_roots=True,matched_affinity='CPU2',wait_for_pa_and_reviewers=True,poll_seconds=60,
              raw_ssd_writes=False,accuracy_evaluated=False)
    print(json.dumps(plan,indent=2),flush=True)
    if args.execute:
        require(os.geteuid()==0,'Run with local sudo')
        subprocess.run(cmd,check=True);write(OUT/'launch.json',plan)


if __name__=='__main__':main()
