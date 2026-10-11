"""Serial preparation and one matched native pair per dataset; preview by default."""
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
from candidates.group_cache_migration_v1.common import OUT,DATA,PY,verify,require,read,write

def blockers():
    from candidates.cache_epoch_v1.start import blockers as original
    active=original()
    result=subprocess.run(['systemctl','list-units','digit-followup-*','--all','--no-pager','--plain','--no-legend'],capture_output=True,text=True,timeout=30,check=True)
    for line in result.stdout.splitlines():
        fields=line.split()
        if len(fields)>=4 and fields[2] in ('active','activating','deactivating') and fields[3]!='exited':active.append(fields[0])
    lease=OUT/'latest_prepare.json'
    if lease.exists():
        v=read(lease);pid=v['controller_pid']
        if pid!=os.getpid() and not v.get('complete') and v.get('stage')!='failed':
            try:
                ticks=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
                if ticks==v['start_ticks']:active.append('migration_preparation_pid_'+str(pid))
            except FileNotFoundError:pass
    return sorted(set(active))

def control(folder,datasets,prepare_only=False):
    require(os.geteuid()==0,'Local sudo required for shared experiment locks and guarded services')
    source=verify();folder.mkdir(parents=True,exist_ok=False);DATA.mkdir(parents=True,exist_ok=True)
    lock_path='/tmp/digit-group-cache-migration-'+('prepare' if prepare_only else 'full')+'-controller.lock'
    with open(lock_path,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        state=dict(passed=False,complete=False,output=str(folder),controller_pid=os.getpid(),start_ticks=ticks,source_sha256=source,datasets={},started_unix=time.time())
        def save(stage,**kw):
            state.update(stage=stage,updated_unix=time.time(),**kw);write(folder/'status.json',state);write(OUT/('latest_prepare.json' if prepare_only else 'latest.json'),state);print(json.dumps(state),flush=True)
        def available():
            while True:
                active=blockers()
                if not active:return
                save('waiting_for_resources',blockers=active);time.sleep(60)
        def run(name,cmd):
            available();require(verify()==source,'Source changed before stage')
            write(folder/(name+'_command.json'),dict(command=cmd))
            with (folder/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:
                    save(name,worker_pid=child.pid,log=str(folder/(name+'.log')),blockers=[])
                    while child.poll() is None:
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:save(name,worker_pid=child.pid)
                except BaseException:
                    # Preparation uses a separate systemd unit. Stop only that
                    # exact unit on cancellation; preserve partial output/logs.
                    if cmd[0]=='/usr/bin/systemd-run':
                        unit=next(x.split('=',1)[1] for x in cmd if x.startswith('--unit='))
                        subprocess.run(['/usr/bin/systemctl','stop','--no-block',unit],check=False)
                    if child.poll() is None:child.terminate()
                    try:child.wait(timeout=180)
                    except subprocess.TimeoutExpired:save(name+'_cleanup_pending',worker_pid=child.pid)
                    raise
                write(folder/(name+'_exit.json'),dict(returncode=child.returncode,finished_unix=time.time()))
                require(child.returncode==0,'Stage failed: '+name)
        def stop(signum,frame):raise InterruptedError('Controller signal '+str(signum))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            if not prepare_only:
                from candidates.group_cache_migration_v1.gpu_check import require_accepted
                try:require_accepted()
                except (RuntimeError,OSError):run('tiny_cuda',[PY,'-B','-u','-m','candidates.group_cache_migration_v1.gpu_check'])
            for ds in datasets:
                available()
                from candidates.group_cache_migration_v1.prepare import check_prepared
                if (DATA/ds/'prepared.json').exists():check_prepared(ds)
                else:
                    unit='digit-group-reuse-'+ds.lower()+'-'+folder.name
                    cmd=['/usr/bin/systemd-run','--wait','--pipe','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),
                         '--property=MemoryMax=128G','--property=MemoryHigh=112G','--property=MemorySwapMax=0',
                         '--property=CPUQuota=400%','--property=Nice=10','--property=IOWeight=10',
                         '--property=KillMode=control-group','--property=Restart=no','--property=UMask=0022',
                         PY,'-B','-u','-m','candidates.group_cache_migration_v1.prepare',ds,'--execute']
                    run(ds+'_prepare',cmd)
                state['datasets'][ds]=dict(prepared=True,native_complete=False);save(ds+'_prepared')
                if prepare_only:continue
                from candidates.group_cache_migration_v1.gpu_check import require_accepted
                try:require_accepted()
                except (RuntimeError,OSError):run(ds+'_tiny_cuda',[PY,'-B','-u','-m','candidates.group_cache_migration_v1.gpu_check'])
                if ds=='IG':
                    run('IG_pair',[PY,'-B','-u','-m','candidates.group_cache_migration_v1.ig.controller','--mode','representative','--model','sage','--gpu','2','--output',str(folder/'IG_pair')])
                    result=read(folder/'IG_pair/comparisons.json');require(result['passed'],'IG pair not accepted')
                else:
                    (OUT/ds/'gpu').mkdir(parents=True,exist_ok=True)
                    run(ds+'_pair',[PY,'-B','-u','-m','candidates.group_cache_migration_v1.'+ds.lower()+'.start','--execute'])
                    candidates=sorted((OUT/ds/'gpu').glob('cache_comparison_*.json'));require(candidates,'Missing comparison')
                    result=read(candidates[-1]);require(result['passed'],'Unaccepted comparison')
                write(folder/(ds+'_summary.json'),result);state['datasets'][ds].update(native_complete=True,summary=str(folder/(ds+'_summary.json')));save(ds+'_complete')
            save('complete',complete=True,passed=True,prepare_only=prepare_only,finished_unix=time.time())
        except BaseException as e:
            save('failed',passed=False,error=repr(e),traceback=traceback.format_exc());raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--control',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--datasets',nargs='+',choices=['IG','UKL','CL'],default=['IG','UKL','CL']);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    require(len(set(a.datasets))==len(a.datasets),'Duplicate dataset')
    if a.control:return control(a.output,a.datasets,a.prepare_only)
    source=verify();checks=read(OUT/'cpu_checks.json');require(checks['passed'] and checks['source_sha256']==source,'Missing current CPU checks')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid());folder=OUT/'runs'/stamp;unit='digit-group-cache-migration-'+stamp
    cmd=['/usr/bin/systemd-run','--unit='+unit,'--service-type=exec','--property=WorkingDirectory='+str(ROOT),
        '--property=KillMode=control-group','--property=Restart=no','--property=MemoryMax=384G','--property=MemorySwapMax=0',
        '--property=UMask=0022','--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PY,'-B','-u',str(Path(__file__).resolve()),'--control','--output',str(folder),'--datasets',*a.datasets]
    if a.prepare_only:cmd.append('--prepare-only')
    cmd.insert(1,'--property=LimitMEMLOCK=infinity')
    value=dict(command=cmd,source_sha256=source,datasets=a.datasets,output=str(folder),unit=unit+'.service',execute=a.execute,
               group_size=2,existing_groups_only=True,strict_group_first=True,complete_profile_epochs=1,profile_seed=23,
               formal_seed=0,matched_roots=True,matched_affinity='CPU2',warmup_batches=20,measured_batches=300,pairs=1,
               wait_for_existing_jobs=True,poll_seconds=60,feature_layout_changed=False,raw_ssd_writes=False,prepare_only=a.prepare_only)
    print(json.dumps(value,indent=2),flush=True)
    if a.execute:
        require(os.geteuid()==0,'Run with local sudo: the shared experiment lock is root-only');subprocess.run(cmd,check=True);write(OUT/('prepare_launch.json' if a.prepare_only else 'launch.json'),value)
if __name__=='__main__':main()
