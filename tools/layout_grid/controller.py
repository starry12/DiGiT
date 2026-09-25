"""Fixed prepared-layout AE replay. No preprocessing, payload writes or old-result reuse."""
import argparse
import csv
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path('/home/embed/digit')
CONTROL=Path('/srv/digit-ae/admin/layout_v1')
OUTPUTS=Path('/srv/digit-ae/layout-results')
PROTOCOLS=ROOT/'results/pa_sage_layout_shared_resume_20260925_v3/protocols'
ORDER=('g2_r20','g1_r00','g1_r80','g4_r00','g4_r80','g2_r00','g2_r80',
       'g1_r10','g1_r20','g1_r40','g2_r10','g2_r40','g4_r10','g4_r20','g4_r40')
NATIVE_SHA='fe9c705882a57da6c25b7978076b598cd984fcba674b8fffb80e8dee84eb2bd3'

def require(ok,message):
    if not ok:raise RuntimeError(message)

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    tmp.replace(p)

def verify_snapshot():
    m=read(CONTROL/'snapshot_manifest.json')
    require(sha(ROOT/'snapshot_identity.json')==sha(CONTROL/'snapshot_identity.json'),'Wrong private namespace')
    for name,digest in m['files'].items():
        p=ROOT/name;require(not p.is_symlink() and sha(p)==digest,'Snapshot drift: '+name)
    require(sha(ROOT/'candidates/pa_sage_layout_shared_resume_v3/manifest.json')==NATIVE_SHA,'Wrong native version')
    require(set(read(PROTOCOLS/'index.json')['points'][i]['point']['id'] for i in range(15))==set(ORDER),'Incomplete grid')
    # Metadata identity only: no large-array hashing or device reads here.
    for name,expected in read(CONTROL/'prepared_identity.json')['files'].items():
        s=Path(name).stat()
        actual=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
        require(actual==expected,'Prepared file changed: '+name)
    return sha(CONTROL/'snapshot_manifest.json')

def validate_point(directory,point):
    state=read(directory/'status.json');report=read(directory/'full_digit_full_accepted.json')
    summary=read(directory/'full_summary.json')
    expected=[('smoke','digit_full',0),('full','digit_full',0)] if point=='g2_r20' else [('full','digit_full',0)]
    require(state['passed'] and state['complete'] and state['stage']=='complete','Incomplete point: '+point)
    require(state['point']['id']==report['point']['id']==point,'Wrong point')
    require([(w['mode'],w['arm'],w['returncode']) for w in state['workers']]==expected,'Missing normal workers')
    require(report['passed'] and not report['smoke'] and not report['source_only'] and report['updates']==1179
            and len(report['epochs'])==1 and report['test'] is None,'Wrong epoch extent')
    require(report['candidate_sha256']==state['candidate_sha256']==NATIVE_SHA,'Wrong native source')
    require(summary['passed'] and summary['report_sha256']['digit_full']==sha(directory/'full_digit_full_accepted.json'),
            'Accepted report changed')
    for mode,_,_ in expected:
        monitor=read(directory/(mode+'_monitor_digit_full')/'external_gpu/summary.json')
        require(monitor['passed'] and monitor['complete'] and monitor['errors']==[],'Monitor failed')
    return sha(directory/'full_digit_full_accepted.json')

def execute(selftest=False):
    require(os.geteuid()==0 and __debug__,'Fixed root service required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==('' if selftest else '2'),'Wrong device visibility')
    os.umask(0o022)
    output=OUTPUTS/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:12]);output.mkdir()
    state=dict(schema='digit-ae-layout-request-v1',passed=False,complete=False,stage='admission',
        started_unix=time.time(),pid=os.getpid(),invocation_id=os.environ.get('INVOCATION_ID'),
        steps=[],completed=[],selftest=selftest,native_acceptance=False,raw_ssd_writes=False,
        reused_full_points=[],prepared_layouts_reused=True,monitoring_backend='nvidia-smi_author_v1')
    def save(**kw):state.update(kw,updated_unix=time.time());write(output/'status.json',state)
    save();write(CONTROL/'state'/('selftest.json' if selftest else 'latest.json'),dict(output=str(output),
        authorized_account='atc27_ae',request_origin='fixed systemd service; no caller identity inferred',
        started_unix=state['started_unix']))
    locks=[];child=None
    def stop(*args):raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for path in ('/run/digit-ae-selfservice/exclusive.lock','/tmp/digit-pa-bidir-controller.lock'):
            handle=open(path,'a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        save(stage='verifying_snapshot_and_prepared_inputs');identity=verify_snapshot();save(snapshot_sha256=identity)
        with (output/'imports.log').open('x') as log:
            subprocess.run([sys.executable,'-I','-B',str(CONTROL/'check_imports.py')],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        if selftest:
            save(stage='complete',complete=True,passed=True,native_acceptance=False,finished_unix=time.time());return
        # No device initialization until occupancy and immutable snapshot checks pass.
        gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        require(gpu=='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703' and gpu not in apps,'GPU 2 changed or occupied')
        (output/'native').mkdir()
        for point in ORDER:
            save(stage='native_'+point)
            cmd=[sys.executable,'-I','-B','-u',str(CONTROL/'point_runner.py'),'--output',str(output/'native'/point)]
            item=dict(point=point,started_unix=time.time(),returncode=None);state['steps'].append(item);save()
            env=dict(os.environ,DIGIT_LAYOUT_PROTOCOL=str(PROTOCOLS/(point+'.json')),DIGIT_LAYOUT_CONTROLLER_PID=str(os.getpid()))
            with (output/(point+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                item['pid']=child.pid;save();rc=child.wait();child=None
            item.update(returncode=rc,finished_unix=time.time());save()
            require(rc==0,'Native point failed: '+point)
            item['accepted_report_sha256']=validate_point(output/'native'/point,point)
            state['completed'].append(point);save()
        sys.path.insert(0,str(ROOT))
        from candidates.pa_sage_layout_shared_resume_v3.aggregate import collect
        summary=collect(read(PROTOCOLS/'index.json'),output/'native')
        require(summary['passed'] and len(summary['points'])==15,'Incomplete aggregation')
        summary.update(origin='fresh AE prepared-layout replay',snapshot_sha256=identity,reused_full_points=[])
        write(output/'summary.json',summary)
        with (output/'summary.csv').open('x') as f:
            w=csv.writer(f);w.writerow(['point','training_seconds','order_excluded_seconds','speedup_vs_g2_r20'])
            for r in summary['points']:w.writerow([r['point']['id'],r['mean_training_epoch_seconds'],r['order_excluded_mean_seconds'],r['speedup_vs_fresh_g2_r20']])
        verify_snapshot()
        save(stage='complete',passed=True,complete=True,native_acceptance=True,summary_sha256=sha(output/'summary.json'),finished_unix=time.time())
    except BaseException as exc:
        save(stage='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for handle in locks:handle.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selftest',action='store_true')
    execute(p.parse_args().selftest)
