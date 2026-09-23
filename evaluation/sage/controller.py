"""Fresh output, paired smoke gate, then the unchanged accepted seed-0 workers."""
import argparse,fcntl,os,signal,subprocess,sys,time
from pathlib import Path
from evaluation.sage.common import *
from evaluation.sage.review import review_worker
from training.sage.run import input_binding
from training.sage.worker import check_inputs
from training.sage.validation import pair_check
from ae.common import check_device
from ae.pa_sage.monitor_control import ExternalMonitor

def plan(mode):
    require(mode in ('smoke','representative'),'Unsupported run mode')
    return [dict(mode=m,arm=a) for m in (('smoke',) if mode=='smoke' else ('smoke','full')) for a in ('gids','digit_full')]

def worker_command(job,output):
    cmd=[sys.executable,'-u','-m','training.sage.worker','--arm',job['arm'],
         '--output',str(output/job['mode']/job['arm']),'--binding',str(output/'inputs.json')]
    return cmd+(['--smoke'] if job['mode']=='smoke' else [])

def arm_evidence(output,job):
    folder=output/job['mode']/job['arm'];monitor=output/(job['mode']+'_monitor_'+job['arm'])
    files=[p for d in (folder,monitor) for p in d.rglob('*') if p.is_file()]
    files.append(output/(job['mode']+'_'+job['arm']+'.log'))
    return {str(p.relative_to(output)):sha(p) for p in sorted(files)}

def run_worker(output,job,state,binding,save,gpu):
    check_device();folder=output/job['mode']/job['arm'];folder.parent.mkdir(exist_ok=True)
    mon_dir=output/(job['mode']+'_monitor_'+job['arm']);mon_dir.mkdir()
    monitor=ExternalMonitor(mon_dir/'external_gpu');child=None
    try:
        ready=monitor.start();cmd=worker_command(job,output)
        w=dict(job,command=cmd,monitor_ready=ready,started_unix=time.time(),status='running');state['workers'].append(w)
        save(stage=job['mode']+'_'+job['arm'])
        with (output/(job['mode']+'_'+job['arm']+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment(gpu));w['pid']=child.pid;save();released=False;deadline=None
            while child.poll() is None:
                receipt=folder/'worker_ready.json'
                if receipt.exists() and not released:
                    require(read(receipt)['passed'],'Worker not ready for acceptance')
                    mon_code=monitor.stop();monitor=None
                    write(folder/'release_worker.json',dict(monitor_stopped=True,monitor_returncode=mon_code));released=True;deadline=time.monotonic()+180
                if deadline is not None:require(time.monotonic()<deadline,'Worker teardown timed out')
                time.sleep(.5)
            code=child.wait();child=None
        w.update(status='complete' if code==0 else 'failed',returncode=code,finished_unix=time.time());save()
        require(code==0 and released,'Worker did not finish normally: '+job['mode']+' '+job['arm'])
        require(mon_code==0,'Monitor process did not finish normally')
        r=review_worker(output,w,state['pid'],binding,gpu)
        accepted=output/(job['mode']+'_'+job['arm']+'_accepted.json');write(accepted,r)
        receipt=dict(passed=True,job=job,report_sha256=sha(folder/'report.json'),accepted_sha256=sha(accepted),files=arm_evidence(output,job))
        write(output/(job['mode']+'_'+job['arm']+'_receipt.json'),receipt)
        return r
    finally:
        try:
            if monitor is not None:monitor.stop()
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:child.kill();child.wait()

def execute(mode,output,gpu):
    # Do not create a result or touch the native device on dry-run (handled by run.py).
    execution=verify();worker_sha=verify_worker();output=output_path(output);output.mkdir(parents=True)
    state=dict(schema='digit-submission-run-v2',mode=mode,dataset='PA',model='sage',seed=0,gpu=gpu,passed=False,complete=False,
               pid=os.getpid(),submission_sha256=execution,worker_candidate_sha256=worker_sha,stage='binding_inputs',workers=[],raw_ssd_writes=False,started_unix=time.time())
    def save(**kw):state.update(kw,updated_unix=time.time());write(output/'status.json',state)
    def stop(signum,frame):raise KeyboardInterrupt('Controller signaled')
    signal.signal(signal.SIGTERM,stop);save()
    try:
        binding=input_binding(output,worker_sha)
        launch=dict(schema='digit-submission-launch-v2',mode=mode,dataset='PA',model='sage',seed=0,gpu=gpu,submission_sha256=execution,
                    worker_candidate_sha256=worker_sha,protocol_sha256=sha(HERE/'pa_sage_protocol.json'),input_binding_sha256=sha(output/'inputs.json'),
                    plan=plan(mode),monitor_policy=POLICY,raw_ssd_writes=False)
        write(output/'launch.json',launch)
        previous=None
        for phase in ('smoke',) if mode=='smoke' else ('smoke','full'):
            if phase=='full':require(previous is not None and previous['passed'],'Full run requires a passing paired smoke')
            reports={}
            for job in [j for j in launch['plan'] if j['mode']==phase]:reports[job['arm']]=run_worker(output,job,state,binding,save,gpu)
            pair=pair_check(reports,phase=='smoke');pair['strict_resource_acceptance']=all(r['external_monitor']['strict_monitor_passed'] for r in reports.values())
            pair['qualification']='complete' if pair['strict_resource_acceptance'] else 'complete_with_monitoring_gaps'
            pair['report_sha256']={a:sha(output/(phase+'_'+a+'_accepted.json')) for a in reports}
            write(output/(phase+'_summary.json'),pair);previous=pair;save(stage=phase+'_accepted')
        require(verify()==execution,'Submission or worker changed during run');check_inputs(binding);check_device()
        save(stage=previous['qualification'],passed=True,complete=True,training_accuracy_io_passed=mode=='representative',
             strict_resource_acceptance=previous['strict_resource_acceptance'],finished_unix=time.time())
    except BaseException as exc:save(stage='failed',passed=False,complete=False,error=type(exc).__name__+': '+str(exc));raise
    if mode=='representative':
        # Export only after all evidence has closed; failure does not re-execute training/test.
        from evaluation.sage.summarize import load_formal,emit
        value=load_formal(output);emit(value,output/'submission_summary')
        save(summary_exported=True)
    return state

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','representative'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    require(__debug__ and os.geteuid()==0 and os.environ.get('TMUX'),'Run in tmux with local sudo, assertions enabled')
    require(a.gpu>=0,'Invalid GPU');os.environ.update(environment(a.gpu))
    with open('/tmp/digit-pa-bidir-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a.mode,a.output,a.gpu)
if __name__=='__main__':main()
