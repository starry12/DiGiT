"""Separate serial controller: original PA workers, alternating paired order."""
import fcntl,signal,subprocess,time
from contextlib import ExitStack
from candidates.pa_sage_bfs256_short_v1.common import *
from candidates.pa_sage_bfs256_short_v1.controller import aggregate
def control(folder,order):
    heavy_gate();source=verify();folder.mkdir(parents=True,exist_ok=False)
    state=dict(passed=False,complete=False,output=str(folder),source_sha256=source,completed=[],started_unix=time.time());child=None
    def save(**kw):
        state.update(kw,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',state)
    def stop(sig,frame):raise InterruptedError('Controller stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with open('/tmp/digit-pa-random-bfs-controller.lock','a+') as unique:
            fcntl.flock(unique,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from candidates.pa_sage_random_bfs_v1.scheduling import blockers
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
                    for arm in order:
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

