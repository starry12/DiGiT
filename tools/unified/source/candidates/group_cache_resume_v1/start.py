import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/group_cache_resume_v1'
PARENT_OUT=ROOT/'results/group_cache_migration_v1'
DATA=Path('/mnt/n0/digit/group_cache_migration_v1')
PY='/home/embed/miniconda3/envs/gids/bin/python'
from ae.common import read,write,require


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    from candidates.group_cache_migration_v1.common import verify as parent
    m=read(HERE/'manifest.json')
    require(parent()==m['parent_sha256'],'Migration source changed')
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Resume source changed: '+name)
    return sha(HERE/'manifest.json')


def dependency(status,active,result,main_status):
    # A free GPU or a stale waiting file alone does not mean IG completed.
    if active in ('active','activating','deactivating','reloading'):return 'waiting'
    require(active in ('inactive','failed'),'Unknown IG service state: '+active)
    require(active=='inactive' and result=='success' and str(main_status)=='0',
            'IG service did not finish successfully; UKL/CL will not start')
    require(status.get('complete') is True and status.get('passed') is True,
            'IG acceptance missing or failed; UKL/CL will not start')
    return 'ready'


def archive_attempt(folder,dest,archive,expected):
    """Only remove three known stale records after verifying rollback copies."""
    require(not dest.exists(),'Partial adjacency exists; preserve it and inspect before resuming')
    present=sorted(p.name for p in folder.iterdir()) if folder.exists() else []
    require(set(present)==set(expected),'Preparation directory changed or has unreviewed output')
    for name,h in expected.items():
        p=folder/name
        require(p.is_file() and not p.is_symlink() and sha(p)==h,'Preparation record changed: '+name)
    if not expected:return
    archive.mkdir(parents=True,exist_ok=False)
    for name,h in expected.items():
        shutil.copy2(folder/name,archive/name)
        require(sha(archive/name)==h,'Preparation rollback copy mismatch')
        with (archive/name).open('rb') as saved:os.fsync(saved.fileno())
    write(archive/'rollback.json',dict(original=str(folder),files=expected,raw_adjacency_existed=False))
    dfd=os.open(str(archive),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(dfd)
    finally:os.close(dfd)
    # Verify every original again before removing any of the old launch records.
    require(all(sha(folder/name)==h for name,h in expected.items()),'Record changed during backup')
    for name in expected:(folder/name).unlink()


def current_dependency(spec):
    require(sha(spec['launch_path'])==spec['launch_sha256'],'IG launch binding changed')
    unit=spec['ig']['unit']+'.service'
    query=subprocess.run(['systemctl','show',unit,'-p','LoadState','-p','ActiveState','-p','Result','-p','ExecMainStatus'],capture_output=True,text=True,timeout=30)
    state=dict(line.split('=',1) for line in query.stdout.splitlines() if '=' in line)
    status_path=Path(spec['ig']['output'])/'status.json'
    status=read(status_path) if status_path.exists() else {}
    if state.get('LoadState')=='not-found':
        # systemd may garbage-collect a successful transient unit. Require both
        # the bound launcher receipt and its accepted experiment result then.
        require(status.get('launcher_sha256')==spec['ig']['launcher_sha256'] and
                status.get('complete') is True and status.get('passed') is True and
                status.get('experiment_result',{}).get('complete') is True and
                status.get('experiment_result',{}).get('passed') is True,
                'IG unit disappeared without successful bound completion')
        return 'ready'
    require(query.returncode==0,'Cannot query IG service: '+query.stderr)
    return dependency(status,state.get('ActiveState','unknown'),state.get('Result'),state.get('ExecMainStatus'))


def control(folder,spec_path):
    require(os.geteuid()==0,'Local sudo required')
    source=verify();spec=read(spec_path)
    require(spec['resume_sha256']==source,'Resume plan source mismatch')
    folder.mkdir(parents=True,exist_ok=False)
    state=dict(passed=False,complete=False,pid=os.getpid(),source_sha256=source,
               output=str(folder),migration_output=str(folder/'migration'),
               ig_dependency=spec['ig'],datasets=['UKL','CL'],poll_seconds=60)
    def save(stage,**kw):
        state.update(stage=stage,updated_unix=time.time(),**kw);write(folder/'status.json',state)
    def stop(signum,frame):raise InterruptedError('Resume signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save('checking_ig')
    try:
        with open('/tmp/digit-group-cache-resume-controller.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while current_dependency(spec)!='ready':
                save('waiting_for_ig');time.sleep(60)
            save('ig_complete')
            # Reuse the original resource checks. No large read or preparation
            # happens before IG success. Protect the stale-record transition
            # with all experiment locks; children acquire their own locks later.
            from candidates.group_cache_migration_v1.start import blockers,control as migrate
            from candidates.uks_group_reuse_v2.common import exclusive
            while True:
                busy=blockers()
                if not busy:break
                save('waiting_for_resources',blockers=busy);time.sleep(60)
            with exclusive():
                require(current_dependency(spec)=='ready','IG dependency became active again')
                from candidates.group_cache_migration_v1.prepare import check_prepared
                for ds in ('UKL','CL'):
                    dest=DATA/ds
                    if (dest/'prepared.json').exists():
                        check_prepared(ds);continue
                    archive_attempt(PARENT_OUT/ds,dest,folder/'prior_preparation'/ds,spec['stale_records'][ds])
            require(verify()==source,'Resume source changed')
            save('migration_running',blockers=[])
            migrate(folder/'migration',['UKL','CL'])
            result=read(folder/'migration/status.json')
            require(result['complete'] and result['passed'],'Migration not accepted')
            save('complete',passed=True,complete=True,result=result,finished_unix=time.time())
    except BaseException as exc:
        save('failed',passed=False,complete=False,error=repr(exc),finished_unix=time.time());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--control',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path);a=p.parse_args()
    if a.control:return control(a.output,a.plan)
    source=verify()
    launch=ROOT/'results/ig_uks_launch_v2/latest.json';ig=read(launch)
    require(ig['unit'].startswith('digit-ig-uks-launch-v2-'),'Unexpected IG dependency')
    from candidates.group_cache_migration_v1.common import verify as parent
    checks=read(PARENT_OUT/'cpu_checks.json')
    require(checks['passed'] and checks['source_sha256']==parent(),'Missing migration CPU checks')
    records={}
    for ds in ('UKL','CL'):
        folder=PARENT_OUT/ds;dest=DATA/ds
        if (dest/'prepared.json').exists():records[ds]={};continue
        require(not dest.exists(),'Incomplete adjacency needs explicit inspection: '+str(dest))
        names=sorted(p.name for p in folder.iterdir()) if folder.exists() else []
        require(set(names)<=set(('source.spec','plan.json','preparation_status.json')),'Unreviewed preparation artifacts: '+ds)
        records[ds]={name:sha(folder/name) for name in names}
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    folder=OUT/'runs'/stamp;plan_path=OUT/'plans'/(stamp+'.json');unit='digit-large-resume-'+stamp
    spec=dict(resume_sha256=source,launch_path=str(launch),launch_sha256=sha(launch),ig=ig,stale_records=records)
    command=['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
             '--property=WorkingDirectory='+str(ROOT),'--property=KillMode=control-group',
             '--property=MemoryMax=384G','--property=MemorySwapMax=0','--property=LimitMEMLOCK=infinity',
             '--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
             '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
             PY,'-B','-u',str(Path(__file__).resolve()),'--control','--output',str(folder),'--plan',str(plan_path)]
    value=dict(unit=unit,output=str(folder),ig_dependency=ig['unit'],datasets=['UKL','CL'],command=command,
               source_sha256=source,poll_seconds=60,prepare_only=False)
    if a.execute:
        require(os.geteuid()==0,'Run with sudo')
        write(plan_path,spec);subprocess.run(command,check=True);write(OUT/'latest.json',value)
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
