#!/usr/bin/python3
"""Root-owned, fixed-protocol AE supervisor. No caller-supplied commands/paths."""
import contextlib,fcntl,hashlib,json,os,re,signal,stat,subprocess,sys,time,uuid
from pathlib import Path
sys.path.insert(0,'/srv/digit-ae/admin/gpu_selection_v1')
import gpu_selection as auto_gpu

CONTROL=Path('/srv/digit-ae/admin/selfservice_v2')
ROOT=Path('/srv/digit-ae/releases/digit_ae_20260923_v3')
NATIVE=ROOT/'results/native'
RUNTIME=Path('/run/digit-ae-selfservice')
STOP=False

def require(ok,message):
    if not ok:raise RuntimeError(message)

def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex);tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def policy():
    p=read(CONTROL/'policy.json')
    if 'DIGIT_AE_GPU_ASSIGNMENT' in os.environ:
        a=auto_gpu.assignment();p.update(gpu=a['index'],gpu_uuid=a['uuid'])
    return p
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()

def trusted(p,symlink=False):
    p=Path(p)
    if p.is_symlink():
        require(symlink,'Unexpected symlink: '+str(p));p=p.resolve(strict=True)
    for q in [p,*p.parents]:
        s=q.stat();require(s.st_uid==0 and not s.st_mode&0o022,'Executable/control path is writable or not root-owned: '+str(q))

def audit():
    p=policy();trusted(CONTROL/'policy.json');trusted(CONTROL/'controller.py');trusted(NATIVE)
    require(str(ROOT)==p['release'],'Unexpected release')
    require(sha(ROOT/'ARTIFACT_MANIFEST.json')==p['package_sha256'],'Release identity changed; administrator must revalidate policy')
    # All code/library search roots must be root-owned and non-writable by users.
    # Avoid following data, reviewer output and historical read-only mounts.
    checked=0
    for tree in (ROOT,Path(p['python']).parents[1]):
        trusted(tree)
        for base,dirs,files in os.walk(tree,followlinks=False):
            if Path(base)==ROOT:dirs[:]=[x for x in dirs if x not in ('data','ssd_state','results','deployment')]
            for name in [*dirs,*files]:
                q=Path(base)/name;s=q.lstat()
                if q.is_symlink():trusted(q,symlink=True)
                else:require(s.st_uid==0 and not s.st_mode&0o022,'Writable/untrusted executable: '+str(q))
                checked+=1
    for name in ('local.json','checks'):
        q=ROOT/'deployment'/name
        if q.exists():trusted(q)
    return checked

def clean_env(job):
    cache=RUNTIME/'cache';scratch=RUNTIME/'scratch'/job.name
    cache.mkdir(exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    return dict(PATH='/srv/digit-ae/env/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',DGLBACKEND='pytorch',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TERM='xterm',
        TMPDIR=str(scratch),XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache/'cuda'),
        TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),MPLCONFIGDIR=str(cache/'matplotlib'),**auto_gpu.propagated())

@contextlib.contextmanager
def file_lock(path):
    fd=os.open(str(path),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        info=os.fstat(fd);require(info.st_uid==0 and stat.S_ISREG(info.st_mode) and not info.st_mode&0o022,'Unsafe lock')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another experiment owns the device/controller lock: '+str(path))
        yield
    finally:os.close(fd)

def gpu_idle(p):
    out=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[x.strip().split(', ') for x in out.splitlines()]
    selected=[x for x in rows if int(x[0])==p['gpu']]
    require(len(selected)==1 and selected[0][1]==p['gpu_uuid'],'Assigned GPU identity changed')
    apps=subprocess.check_output(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    require(not any(line.startswith(p['gpu_uuid']+',') for line in apps.splitlines()),'Assigned GPU has an active compute process')
    require(int(selected[0][2])<=p['idle_gpu_memory_mib_limit'],'Assigned GPU memory is not idle')
    device=subprocess.run(['/usr/bin/fuser','/dev/libnvm0'],capture_output=True,text=True,timeout=10)
    require(device.returncode==1 and not device.stdout.strip(),'SSD controller is busy or unavailable')

def key_valid(key,action,p):
    require((key=='selftest' and action=='selftest') or (key in p['pairs'] and action in p['actions']) or (key=='ig-all' and action=='run'),'Unsupported fixed experiment')

def execution_pairs(key,action,p):
    key_valid(key,action,p)
    if key=='ig-all':
        require(p.get('queue')==['ig-sage','ig-gcn','ig-gat'],'Unexpected IG queue')
        return p['queue']
    return [key]

def run_serial(pairs,callback):
    # Propagate failures immediately: later models must not silently run.
    completed=[]
    for pair in pairs:
        callback(pair)
        completed.append(pair)
    return completed

def probe(key):
    p=policy();key_valid(key,'smoke',p);sys.path.insert(0,str(ROOT))
    from artifact import _deployment,_preflight
    from artifact_integrity import verify_package
    from argparse import Namespace
    digest=verify_package();require(digest==p['package_sha256'],'Wrong package');_deployment(digest)
    dataset,model=p['pairs'][key]
    receipt=ROOT/'deployment/checks'/(dataset+'_'+model+'.json')
    try:
        if receipt.exists():
            trusted(receipt);folder=ROOT/read(receipt)['output'];folder.relative_to(NATIVE);trusted(folder)
        _preflight(Namespace(dataset=dataset,model=model,gpu=p['gpu']),digest)
    except (RuntimeError,FileNotFoundError,ValueError,KeyError):return 3
    return 0

def stopped(signum,frame):
    global STOP
    STOP=True
    raise KeyboardInterrupt('Service stopped')

def worker(job):
    job=Path(job);require(job.parent==NATIVE and re.fullmatch(r'selfservice_[0-9]{8}_[0-9]{6}_[a-z-]+_[0-9a-f]{12}',job.name),'Invalid managed job path');trusted(job)
    spec=read(job/'job.json');p=policy();key=spec['pair'];action=spec['action'];key_valid(key,action,p)
    if key!='selftest':
        a=auto_gpu.assignment();require(spec['gpu']==a['index'] and spec['gpu_uuid']==a['uuid'],'Worker GPU differs')
        require(spec['gpu_selection']['transport_sha256']==auto_gpu.transport(),'GPU transport changed')
    state=dict(spec,stage='starting',passed=False,complete=False,native_acceptance=False)
    def save(stage,**kw):state.update(stage=stage,updated_unix=time.time(),**kw);write(job/'status.json',state);print(json.dumps(dict(stage=stage,**kw)),flush=True)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    log=(job/'launcher.log').open('a',buffering=1);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    env=clean_env(job);require(os.environ.get('TMUX'),'Private tmux session missing');env['TMUX']=os.environ['TMUX']
    python=p['python'];active_pair=key
    def invoke(action_name,output):
        require(not output.exists(),'Output already exists')
        cmd=[python,'-u','-B',str(ROOT/'artifact.py'),action_name,'--output',str(output)]
        if key!='selftest':cmd+=['--dataset',p['pairs'][active_pair][0],'--model',p['pairs'][active_pair][1],'--gpu',str(p['gpu'])]
        child=subprocess.Popen(cmd,cwd=str(ROOT),env=env)
        try:code=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        require(code==0,'Artifact action failed with exit '+str(code)+'; inspect '+str(output))
        receipt=read(output/'artifact_invocation.json');require(receipt['exit_code']==0,'Artifact receipt failed')
        result=read(output/('report.json' if action_name=='example' else 'status.json'))
        require(result.get('passed') is True and (action_name=='example' or result.get('complete') is True),'Artifact result is not accepted')
        return result
    try:
        count=audit();save('trusted_code_checked',executable_paths_checked=count)
        if key=='selftest':
            invoke('example',job/'cpu_example');save('cpu_passed_wait',cpu_example_passed=True)
            # Bounded interval permits the installer to verify graceful stop.
            time.sleep(20);save('complete',passed=True,complete=True);return 0
        completed=[];summaries={}
        def execute_pair(pair):
            nonlocal active_pair
            active_pair=pair
            folder=job/pair if key=='ig-all' else job
            if key=='ig-all':folder.mkdir()
            save('pair_starting',current_pair=pair,completed_pairs=list(completed))
            # Original preflight locks stay shared with all native controllers.
            with contextlib.ExitStack() as stack:
                for path in p['shared_locks']:stack.enter_context(file_lock(path))
                gpu_idle(p)
                check=subprocess.run([python,'-I','-B',str(CONTROL/'controller.py'),'probe',pair],cwd=str(ROOT),env=env)
                require(check.returncode in (0,3),'Deployment/integrity check failed')
                if action=='check' or check.returncode==3:
                    save('preflight');invoke('check',folder/'check')
                else:save('preflight_reused')
            if action=='check':completed.append(pair);return
            gpu_idle(p);save('smoke' if action=='smoke' else 'representative')
            report=invoke('smoke' if action=='smoke' else 'representative',folder/'experiment')
            summaries[pair]=dict(experiment=str(folder/'experiment'),stage=report.get('stage'),summary=str(folder/'experiment/submission_summary') if action=='run' else None)
            completed.append(pair);save('pair_complete',completed_pairs=list(completed))
        run_serial(execution_pairs(key,action,p),execute_pair)
        if key=='ig-all':write(job/'queue_summary.json',dict(passed=True,package_sha256=p['package_sha256'],order=completed,experiments=summaries,accuracy_claim=False,epoch_time_claim=False))
        save('complete',passed=True,complete=True,native_acceptance=action!='check',completed_pairs=completed,experiments=summaries)
        return 0
    except BaseException as exc:
        save('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=type(exc).__name__+': '+str(exc),passed=False,complete=False)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1

def supervise(key,action):
    p=policy();key_valid(key,action,p);trusted(NATIVE);trusted(CONTROL/'state/latest')
    job=NATIVE/('selfservice_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'_'+key+'_'+uuid.uuid4().hex[:12]);job.mkdir(mode=0o755)
    spec=dict(pair=key,action=action,gpu=None,gpu_uuid=None,package_sha256=p['package_sha256'],output=str(job),started_unix=time.time())
    write(job/'job.json',spec);write(job/'status.json',dict(spec,stage='request_received',passed=False,complete=False,native_acceptance=False))
    write(CONTROL/'state/latest'/(key+'-'+action+'.json'),spec)
    print('Managed job:',job,flush=True)
    socket=RUNTIME/(job.name+'.sock');tmux=['/usr/bin/tmux','-S',str(socket),'-f','/dev/null']
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        with file_lock(RUNTIME/'exclusive.lock'), contextlib.ExitStack() as gpu_stack:
            if key!='selftest':
                selection=auto_gpu.activate(gpu_stack,max_used_mib=p['idle_gpu_memory_mib_limit'])
                p=policy();spec.update(gpu=p['gpu'],gpu_uuid=p['gpu_uuid'],gpu_selection=selection)
                write(job/'gpu_selection.json',selection)
                write(job/'job.json',spec)
                write(job/'status.json',dict(spec,stage='gpu_selected',passed=False,complete=False,native_acceptance=False))
                write(CONTROL/'state/latest'/(key+'-'+action+'.json'),spec)
            write(CONTROL/'state/latest'/(key+'.json'),spec)
            env=clean_env(job)
            subprocess.run(tmux+['new-session','-d','-s','ae','-c',str(ROOT),'/usr/bin/python3','-I','-B',str(CONTROL/'controller.py'),'worker',str(job)],check=True,env=env,cwd=str(ROOT))
            deadline=time.monotonic()+48*3600
            while True:
                status=read(job/'status.json')
                if status['stage'] in ('complete','failed','interrupted'):
                    for _ in range(50):
                        if subprocess.run(tmux+['has-session','-t','ae'],capture_output=True,env=env).returncode!=0:
                            return 0 if status.get('passed') else 1
                        time.sleep(.1)
                    raise RuntimeError('Worker did not exit after closing its status')
                alive=subprocess.run(tmux+['has-session','-t','ae'],capture_output=True,env=env).returncode==0
                require(alive,'Private worker ended without a closed status')
                require(time.monotonic()<deadline,'Experiment exceeded 48 hours')
                time.sleep(1)
    except BaseException as exc:
        state=read(job/'status.json');state.update(stage='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',passed=False,complete=False,error=type(exc).__name__+': '+str(exc),updated_unix=time.time());write(job/'status.json',state)
        if socket.exists():subprocess.run(tmux+['send-keys','-t','ae','C-c'],capture_output=True)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        # This socket belongs only to this managed unit, never the author's tmux.
        if socket.exists():subprocess.run(tmux+['kill-server'],capture_output=True)

def main():
    require(os.geteuid()==0,'This helper is only executed by the fixed system services')
    os.umask(0o022)
    args=sys.argv[1:]
    if len(args)==3 and args[0]=='supervise':return supervise(args[1],args[2])
    if len(args)==2 and args[0]=='worker':return worker(args[1])
    if len(args)==2 and args[0]=='probe':return probe(args[1])
    raise RuntimeError('Unsupported internal invocation')
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:print(type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
