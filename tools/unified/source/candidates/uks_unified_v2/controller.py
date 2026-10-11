"""Hold shared locks through fixtures and all fresh paired workers."""
import fcntl,os,signal,subprocess,sys,time
from contextlib import ExitStack
from .common import *

def execute(mode,folder):
    from candidates.uks_pa_fast_v1.common import LOCKS
    from candidates.uks_group_compare_v2.resources import target_gpu_busy
    from candidates.pa_sage_random_bfs_full20_v1.scheduling import blockers
    from candidates.unified_training_v2.series import run_series
    from candidates.unified_results_v2.report import summarize
    from candidates.unified_results_v2.__main__ import export_csv
    source=verify();c=config(mode);folder.mkdir(parents=True,exist_ok=False)
    write(folder/'config.json',c);child=None;state=dict(passed=False,complete=False,source_sha256=source)
    def save(**kw):state.update(kw,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',dict(state,output=str(folder)))
    def stop(sig,frame):raise InterruptedError('UKS unified controller signaled')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while True:
            busy=blockers()
            if busy:save(stage='waiting',blockers=busy);time.sleep(60);continue
            stack=ExitStack();files=[]
            try:
                for path in LOCKS:
                    require(Path(path).is_file() and not Path(path).is_symlink(),'Unsafe lock')
                    f=stack.enter_context(open(path,'rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);files.append(f)
            except BlockingIOError:stack.close();save(stage='waiting',blockers=['shared locks']);time.sleep(60);continue
            except BaseException:stack.close();raise
            break
        with stack:
            fds=tuple(f.fileno() for f in files)
            def run(arm,rep,fixtures=False):
                nonlocal child
                from candidates.uks_pa_fast_v1.monitor import snapshot
                require(source==verify(),'Source changed');require(not target_gpu_busy(),'GPU2 occupied')
                name='gpu_checks' if fixtures else 'r%d_%s'%(rep,arm)
                cmd=[sys.executable,'-B','-u','-m','candidates.uks_unified_v2.worker','--mode',mode,'--arm',arm,
                    '--repetition',str(rep),'--output',str(folder/name),'--lock-fds',','.join(map(str,fds))]
                if fixtures:cmd+=['--fixtures']
                write(folder/(name+'_command.json'),dict(command=cmd))
                with (folder/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=fds)
                    save(stage=name,worker_pid=child.pid)
                    while child.poll() is None:
                        with (folder/'host_observations.jsonl').open('a') as f:
                            f.write(__import__('json').dumps(dict(stage=name,**snapshot()))+'\n')
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:save(stage=name,worker_pid=child.pid)
                    code=child.returncode;child=None
                write(folder/(name+'_exit.json'),dict(returncode=code))
                require(code==0,'UKS worker failed: '+name)
                deadline=time.monotonic()+60
                while target_gpu_busy() and time.monotonic()<deadline:time.sleep(1)
                require(not target_gpu_busy(),'GPU ownership not released')
                report=read(folder/name/'report.json');require(report['passed'],'Worker not accepted')
                require(source==verify(),'Source changed')
                return report
            run('digit',1,True)
            def launch(config,arm,rep):
                r=run(arm,rep)
                require(r['final']['feature_checks']==4,'Missing native feature checks')
                return dict(report=r,returncode=0,cleanup_verified=True,admission_verified=True,correctness_verified=True)
            run_series(c,launch,lambda s:write(folder/'series.json',s))
            result=summarize(c,read(folder/'series.json')['receipts'])
            write(folder/'summary.json',result);export_csv(folder/'summary.csv',result)
            save(stage='complete',passed=True,complete=True,worker_pid=None)
            return folder/'summary.json'
    except BaseException as e:save(stage='failed',error=repr(e));raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
