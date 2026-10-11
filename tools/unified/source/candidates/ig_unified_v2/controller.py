"""Native pairs using the existing monitor lifecycle and shared aggregation."""
import os,subprocess,sys,time
from pathlib import Path
from .common import *
from .review import review_worker
from candidates.ig_demand_pair_v1.monitor_control import ExternalMonitor
from ae.common import check_device

def worker_command(job,output):
    cmd=[sys.executable,'-B','-u','-m','candidates.ig_unified_v2.worker',
         '--scope',job['scope'],'--arm',job['arm'],'--repetition',str(job['repetition']),
         '--output',str(output/job['mode']/job['arm']),'--launch',str(output/'launch.json')]
    return cmd+(['--smoke'] if job['smoke'] else [])

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
                if monitor is not None:require(monitor.process.poll() is None,'External NVML monitor exited during worker; inspect monitor log and summary')
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
        check_device()
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

def execute(scope,output,gpu=2):
    from candidates.unified_training_v2.config import load_config
    from candidates.unified_training_v2.series import run_series
    from .bfs import main as unused
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=verify();state=dict(pid=os.getpid(),workers=[],stage='preparation',passed=False,complete=False,source_sha256=source)
    def save(**kw):state.update(kw,updated_unix=time.time());write(output/'status.json',state)
    c=load_config('IG',scope)
    write(output/'config.json',c)
    from candidates.ig_demand_pair_v1.inputs import bind as bind_reference
    reference=bind_reference(output,source)
    reference['note']='Legacy graph/feature/label/SSD identity snapshot only. Runtime roots use the new BFS receipt; DiGiT cache uses the new full BFS profile.'
    write(output/'inputs.json',reference);check_inputs(reference)
    bfs=ROOT/'data/ig_unified_v2/bfs.npy';bfs_receipt=bfs.with_suffix('.json')
    if not bfs_receipt.exists():
        subprocess.run([sys.executable,'-B','-m','candidates.ig_unified_v2.bfs','--output',str(bfs),'--execute'],check=True,cwd=ROOT)
    require(read(bfs_receipt)['passed'],'BFS preparation failed')
    launch=dict(controller_pid=os.getpid(),source_sha256=source,bfs_receipt=str(bfs_receipt),
                profile_directory=str(output/'profile'),scope=scope,reference_input_snapshot=str(output/'inputs.json'),
                reference_input_snapshot_sha256=sha(output/'inputs.json'))
    write(output/'launch.json',launch);save()
    try:
        smokes={}
        for arm in ('gids','digit_full'):
            job=dict(mode='smoke_'+arm,scope=scope,arm=arm,smoke=True,repetition=1,model='sage')
            smokes[arm]=run_worker(output,job,state,launch,save,gpu)
        require(smokes['gids']['initial']['initial_model_sha256']==smokes['digit_full']['initial']['initial_model_sha256'],'Unmatched initial model')
        def launch_worker(config,arm,repetition):
            name='digit_full' if arm=='digit' else 'gids'
            job=dict(mode='round%d_%s'%(repetition,name),scope=scope,arm=name,smoke=False,repetition=repetition,model='sage')
            report=run_worker(output,job,state,launch,save,gpu)
            return dict(report=report,returncode=0,cleanup_verified=True,admission_verified=True,correctness_verified=True)
        result=run_series(c,launch_worker,lambda s:write(output/'series.json',s))
        require(source==verify(),'Source changed');check_inputs(reference);check_device()
        from .report import export
        result=export(output,c,read(output/'series.json'))
        save(stage='complete',passed=True,complete=True,result=result)
    except BaseException as exc:
        save(stage='failed',passed=False,complete=False,error=repr(exc));raise
    return state
