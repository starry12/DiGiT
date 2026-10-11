"""Serialize short and full mixed-read/BFS workers under existing device locks."""
import argparse,fcntl,signal,subprocess,traceback
from .common import *
from candidates.pa_sage_mixed_group_v1.queue import exclusion
from candidates.pa_sage_512b_pair_v4.control import gpu_admission


def summarize(reports):
    require([r['variant'] for r in reports]==list(VARIANTS),'Incomplete comparison')
    for r in reports:validate_completion(r)
    gids,a,b=reports
    require(all(not r['smoke'] for r in reports),'Smoke is not performance')
    require(len({r['initial_model_sha256'] for r in reports})==1,'Unpaired model initialization')
    require(a['cpu_cache']==b['cpu_cache'] and gids['cpu_cache']['rows']==a['cpu_cache']['rows'],'CPU feature capacities differ')
    require([e['root_order_sha256'] for e in gids['epochs']]==[e['root_order_sha256'] for e in a['epochs']],'GIDS/random roots differ')
    rows={}
    for r in reports:
        epochs=r['epochs'];commands=sum(e['device_io']['completed_commands'] for e in epochs)
        active=sum(e['timing']['ssd_active_seconds'] for e in epochs)
        elapsed=sum(e['timing']['e2e_seconds'] for e in epochs)
        rows[r['variant']]=dict(epochs=len(epochs),training_seconds=elapsed,mean_epoch_seconds=elapsed/len(epochs),
            epoch_seconds=[e['timing']['e2e_seconds'] for e in epochs],
            validation_accuracy=[e['validation']['accuracy'] for e in epochs],
            final_validation_accuracy=epochs[-1]['validation']['accuracy'],test_accuracy=r['test']['accuracy'],
            ssd_active_seconds=active,ssd_commands=commands,ssd_completed_bytes=sum(e['device_io']['completed_bytes'] for e in epochs),
            active_iops=commands/active if active else None,
            primary_512b_commands=sum(e['pair_counts'][0] for e in epochs),primary_1kib_commands=sum(e['pair_counts'][1] for e in epochs),
            cpu_service_fraction=sum(e['feature_access']['cpu'] for e in epochs)/sum(e['logical_feature_requests'] for e in epochs))
    for variant in VARIANTS:rows[variant]['speedup_vs_gids']=rows['gids_decay']['training_seconds']/rows[variant]['training_seconds']
    return dict(passed=True,complete=True,rows=rows,bfs_speedup_vs_random=rows['random_decay']['training_seconds']/rows['bfs64_decay']['training_seconds'],
        limits=['Single seed; directed graph; retained GIDS 512B baseline and optimized mixed-read DiGiT',
                'All three arms use lr 0.01 -> 0.005 -> 0.0025, weight decay 0.001',
                'Equal CPU feature capacity: GIDS per-node RevPR versus unchanged DiGiT frequency hot set',
                'Training includes order iterator and read planning, excludes validation/test/setup',
                'Evaluation uses fixed directed traces and original CPU features, without native cache pollution',
                '20-epoch warm-cache comparison is separate from historical one-epoch timings'])


def run(folder):
    require(os.geteuid()==0,'Native controller needs local sudo')
    singleton=open('/tmp/digit-pa-mixed-bfs-v2-controller.lock','a+')
    fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder.mkdir(parents=True,exist_ok=False)
    state=dict(passed=False,complete=False,pid=os.getpid(),output=str(folder),completed_stages=[],started_unix=time.time())
    def save(**kw):
        state.pop('reason',None);state.update(kw,updated_unix=time.time())
        write(folder/'status.json',state);write(OUT/'latest.json',state);print(json.dumps(state),flush=True)
    def stop(signum,frame):raise KeyboardInterrupt('Controller signaled')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);child=None
    try:
        save(stage='queued')
        with exclusion(save) as fd:
            source=verify();check_inputs(read(OUT/'inputs.json'));checks=read(OUT/'cpu_checks.json')
            require(checks['passed'] and checks['source_sha256']==source,'Missing/stale CPU checks')
            gpu_admission(folder,'gpu_check');save(stage='gpu_only_checks')
            with (folder/'gpu_checks.log').open('x') as log:
                subprocess.run([PY,'-B','-u','-m',MODULE+'.gpu_check','--output',str(folder/'gpu_checks.json')],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            require(read(folder/'gpu_checks.json')['passed'],'GPU lifecycle check failed')
            reports=[]
            for smoke in (True,False):
                for variant in VARIANTS:
                    stage=('smoke_' if smoke else 'full_')+variant;dest=folder/stage
                    gpu_admission(folder,stage)
                    cmd=[PY,'-B','-u','-m',MODULE+('.gids_worker' if variant=='gids_decay' else '.worker'),'--output',str(dest),'--lock-fd',str(fd)]
                    if variant!='gids_decay':cmd+=['--variant',variant]
                    if smoke:cmd.append('--smoke')
                    write(folder/(stage+'_command.json'),dict(command=cmd))
                    with (folder/(stage+'.log')).open('x') as log:
                        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
                        save(stage=stage,worker_pid=child.pid,worker_output=str(dest))
                        code=child.wait();child=None
                    write(folder/(stage+'_exit.json'),dict(returncode=code,finished_unix=time.time()))
                    require(code==0,'Worker failed: '+stage)
                    a=read(dest/'accepted.json');r=read(dest/'report.json')
                    require(a['passed'] and a['report_sha256']==sha(dest/'report.json'),'Report lacks acceptance')
                    require(r['candidate_sha256']==source and r['variant']==variant and r['smoke']==smoke,'Wrong worker/source')
                    validate_completion(r)
                    if not smoke:reports.append(r)
                    state['completed_stages'].append(stage);save(stage=stage+'_accepted')
            result=summarize(reports)
            require(verify()==source,'Candidate changed');check_inputs(read(OUT/'inputs.json'))
            write(folder/'summary.json',result);write(OUT/'summary.json',result)
            save(stage='complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        save(stage='failed',error=str(error),finished_unix=time.time());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
