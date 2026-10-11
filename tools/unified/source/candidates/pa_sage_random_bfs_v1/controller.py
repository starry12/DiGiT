"""Serial bounded preparation/profile/smoke/pair; no automatic failure retry."""
import argparse
import fcntl
import signal
import subprocess
import time
from contextlib import ExitStack
from .common import *

def aggregate(reports):
    require(set(reports)=={'gids','digit'},'Missing arm')
    for arm,r in reports.items():
        require(r['passed'] and not r['smoke'] and r['arm']==arm and r['updates']==320 and r['measured_batches']==300,'Incomplete arm')
        require(r['training']['reconciled'] and r['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'Counters/cold cache failed')
        require(r['protocol']==cfg() and r['source_sha256']==verify(),'Wrong configuration')
    a,b=reports['gids'],reports['digit']
    require(a['initial_model_sha256']==b['initial_model_sha256'],'Unmatched initial model')
    from .dataset import check_report_roots,comparison_roots
    for arm,r in reports.items():check_report_roots(r,arm)
    root_comparison=comparison_roots()
    for k in ('cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes'):require(a['cache'][k]==b['cache'][k],'Unequal feature cache capacity')
    require(b['pipeline']['submitted']==b['pipeline']['collected']==320 and b['pipeline']['outstanding']==0,'Incomplete async pipeline')
    rows={}
    for arm,r in reports.items():
        windows=r['windows'][1:];groups=sum(w['group_sampled_edges'] for w in windows);edges=sum(w['outer_sampled_edges'] for w in windows)
        ssd=r['training']['serving']['ssd_served_rows'];pairs=r['training']['request_sizes']['primary_1k']
        rows[arm]=dict(seconds=r['measured_seconds'],serving=r['training']['serving'],device=r['training']['device'],
            request_sizes=r['training']['request_sizes'],group_sampled_edge_fraction=groups/edges if edges else 0.,
            ssd_grouped_row_fraction=2*pairs/ssd if ssd else 0.,setup_seconds=r['setup_seconds'])
    return dict(passed=True,complete=True,arms=rows,speedup=a['measured_seconds']/b['measured_seconds'],
        root_comparison=root_comparison,aggregation='one random-GIDS versus BFS-DiGiT pair, all 300 timed batches retained',accuracy_evaluated=False,
        stable_speedup_established=False,full_epoch_training=False,
        limitations=['Different root orders and short-window examples; system comparison includes ordering, cache, sampling and execution differences.','Actual PA features/labels but no validation/test in this short pair.',
                    'One short comparison does not establish paper performance or accuracy.'])

def control(folder):
    heavy_gate();source=verify();folder.mkdir(parents=True,exist_ok=False)
    state=dict(passed=False,complete=False,output=str(folder),source_sha256=source,completed=[],started_unix=time.time());child=None
    def save(**kw):
        state.update(kw,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',state)
    def stop(sig,frame):raise InterruptedError('Controller stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with open('/tmp/digit-pa-random-bfs-controller.lock','a+') as unique:
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
                    if module in ('worker','profile'):
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
                reports={}
                for smoke in (True,False):
                    for arm in cfg()['ablation_order']:
                        name=('smoke_' if smoke else '')+arm
                        run(name,'worker',['--arm',arm,'--output',str(folder/name)]+(['--smoke'] if smoke else []))
                        r=read(folder/name/'report.json')
                        if smoke:require(r['passed'] and r['feature_bit_exact'] and r['graph_edges_checked'] and r['feature_checks']==24,'Real feature/graph smoke failed')
                        else:reports[arm]=r
                result=aggregate(reports);write(folder/'summary.json',result);write(OUT/'summary.json',result)
                save(stage='complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as e:save(stage='failed',error=repr(e));raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);control(p.parse_args().output)
