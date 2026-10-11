"""Serial bounded preparation/profile/smoke/pair; no automatic failure retry."""
import argparse
import fcntl
import signal
import subprocess
import time
from contextlib import ExitStack
from .common import *

def aggregate(reports,accuracy):
    from .dataset import validate_epochs,check_report_roots
    require(set(reports)==set(accuracy)=={'gids','digit'},'Missing arm/evaluation')
    for arm,r in reports.items():
        validate_epochs(r);check_report_roots(r,arm)
        require(r['protocol']==cfg() and r['source_sha256']==verify(),'Configuration changed')
        require(accuracy[arm]['source_sha256']==verify() and accuracy[arm]['arm']==arm,'Wrong evaluation provenance')
        require([v['model_sha256'] for v in accuracy[arm]['validation']]==[e['model_sha256'] for e in r['epochs']] and accuracy[arm]['test']['model_sha256']==r['final_model_sha256'],'Evaluation used wrong checkpoint')
        require(accuracy[arm]['passed'] and len(accuracy[arm]['validation'])==20 and accuracy[arm]['test']['epoch']==20,'Incomplete evaluation')
    a,b=reports['gids'],reports['digit']
    require(a['initial_model_sha256']==b['initial_model_sha256'],'Unmatched initialization')
    for key in ('cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes'):require(a['cache'][key]==b['cache'][key],'Unequal cache capacity')
    arms={}
    for arm,r in reports.items():
        arms[arm]=dict(training_seconds=r['measured_seconds'],mean_epoch_seconds=r['measured_seconds']/20,
            epoch_seconds=[e['seconds'] for e in r['epochs']],training=r['training'],accuracy=accuracy[arm],
            setup_seconds=r['setup_seconds'])
    return dict(passed=True,complete=True,arms=arms,speedup=a['measured_seconds']/b['measured_seconds'],
        aggregation='ratio of complete 20-epoch training totals, including cold epoch1 and all tails',
        full_epoch_training=True,accuracy_evaluated=True,matched_full_training_set=True,matched_root_order=False,
        stable_speedup_established=False,limitations=['One seed and one pair; ordering is part of this system comparison.',
        'Validation evaluated after training from each saved checkpoint; test uses epoch20, not best validation.',
        'Accuracy evaluation uses common frozen neighbor traces and original features, separate from timed SSD training.'])

def control(folder):
    heavy_gate();source=verify();folder.mkdir(parents=True,exist_ok=False)
    state=dict(passed=False,complete=False,output=str(folder),source_sha256=source,completed=[],started_unix=time.time());child=None
    def save(**kw):
        state.update(kw,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',state)
    def stop(sig,frame):raise InterruptedError('Controller stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with open('/tmp/digit-pa-random-bfs-full20-controller.lock','a+') as unique:
            fcntl.flock(unique,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from .scheduling import blockers
            while True:
                busy=blockers()
                if busy:save(stage='waiting',blockers=busy,poll_seconds=60);time.sleep(60);continue
                stack=ExitStack();files=[]
                try:
                    for path in LOCKS:
                        require(Path(path).is_file() and not Path(path).is_symlink(),'Unsafe experiment lock')
                        f=stack.enter_context(open(path,'rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);files.append(f)
                except BlockingIOError:stack.close();save(stage='waiting',blockers=['experiment lock busy']);time.sleep(60);continue
                except BaseException:stack.close();raise
                break
            with stack:
                fds=tuple(f.fileno() for f in files);lockarg=','.join(map(str,fds))
                def run(name,module,args):
                    nonlocal child
                    require(source==verify(),'Source changed')
                    if module in ('worker','profile','evaluate'):
                        from candidates.uks_group_compare_v2.resources import target_gpu_busy
                        require(not target_gpu_busy(),'GPU2 occupied before '+name)
                    cmd=[PY,'-B','-u','-m',MODULE+'.'+module,*args,'--lock-fds',lockarg]
                    write(folder/(name+'_command.json'),dict(command=cmd))
                    with (folder/(name+'.log')).open('x') as log:
                        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=fds)
                        save(stage=name,worker_pid=child.pid,blockers=[])
                        from candidates.uks_pa_fast_v1.monitor import snapshot
                        while child.poll() is None:
                            with (folder/'host_observations.jsonl').open('a') as f:f.write(json.dumps(dict(stage=name,**snapshot()))+'\n')
                            try:child.wait(timeout=5)
                            except subprocess.TimeoutExpired:save(stage=name,worker_pid=child.pid)
                        code=child.returncode;child=None
                    write(folder/(name+'_exit.json'),dict(returncode=code,finished_unix=time.time()))
                    require(code==0,'Stage failed: '+name)
                    state['completed'].append(name);save(stage=name+'_complete',worker_pid=None)
                require((DATA/'prepared.json').exists(),'Accepted parent preparation required')
                prepared()
                run('gpu_checks','worker',['--fixtures','--output',str(folder/'gpu_checks')])
                require((DATA/'profile.json').exists(),'Accepted unchanged DiGiT BFS profile required')
                from candidates.pa_sage_uks_style_v1.provenance import check_prepared_source
                prof=read(DATA/'profile.json');require(prof['passed'],'Incomplete profile')
                check_prepared_source(prof['source_sha256']);check_binding(prof['hot'])
                save(reused_preparation_source_sha256=prepared()['source_sha256'],reused_profile_source_sha256=prof['source_sha256'])
                reports={};accuracy={}
                for smoke in (True,False):
                    for arm in cfg()['ablation_order']:
                        name=('smoke_' if smoke else '')+arm
                        run(name,'worker',['--arm',arm,'--output',str(folder/name)]+(['--smoke'] if smoke else []))
                        r=read(folder/name/'report.json')
                        if smoke:require(r['passed'] and r['feature_bit_exact'] and r['graph_edges_checked'] and r['feature_checks']==24,'Real feature/graph smoke failed')
                        else:
                            reports[arm]=r
                            run('evaluate_'+arm,'evaluate',['--arm',arm,'--training',str(folder/name),'--output',str(folder/('evaluate_'+arm))])
                            accuracy[arm]=read(folder/('evaluate_'+arm)/'report.json')
                result=aggregate(reports,accuracy);write(folder/'summary.json',result);write(OUT/'summary.json',result)
                save(stage='complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as e:save(stage='failed',error=repr(e));raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);control(p.parse_args().output)
