"""Plan by default. Explicit --execute performs ONE bounded smoke, no retries."""
import argparse,datetime,fcntl,hashlib,json,os,pwd,re,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ukl_memory_ladder_20261002_v5'
# Direct script launch needs the project package root.
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from candidates.ukl_memory_ladder_v5.protocol import validate_tier,prerequisite,budget,DATA
from candidates.ukl_memory_ladder_v5.observer import Observer
OBSERVER=None
PY='/home/embed/miniconda3/envs/gids/bin/python'
BAD=re.compile(r'Xid|NVRM.*(?:error|fail)|ext4.*(?:error|warning)|does not have buffers|BUG:|WARNING:|Call Trace:|blocked for more than|I/O error|corrupt',re.I)

def run(args,timeout=10):
    return subprocess.run(args,capture_output=True,text=True,check=True,timeout=timeout).stdout

def snapshot():
    raw=Path('/proc/meminfo').read_text();d={}
    for name in ('MemAvailable','Dirty','Writeback'):
        d[name]=int(re.search(r'^'+name+r':\s+(\d+)',raw,re.M)[1])*1024
    d['time']=time.time()
    if OBSERVER is not None:OBSERVER.sample(d)
    return d

def clean_memory(before,after):
    return after['MemAvailable']>=64*1024**3 and after['Dirty']<2*1024**3 and after['Dirty']-before['Dirty']<256*1024**2 and after['Writeback']<256*1024**2

def select_gpu(gpus,apps):
    busy={x.strip() for x in apps.splitlines() if x.strip()};idle=[]
    for row in gpus.splitlines():
        fields=[x.strip() for x in row.split(',')]
        if len(fields)!=4:raise RuntimeError('Unexpected GPU query format')
        idx,uuid,mem,util=fields
        if uuid not in busy and int(mem)<=64 and int(util)==0:idle.append((int(idx),uuid))
    if not idle:raise RuntimeError('No idle GPU; will not queue/retry')
    return next((x for x in idle if x[0]==2),idle[0])

def gpu():
    return select_gpu(run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),run(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits']))

def command(unit,out,uuid,tier=65536):
    b=budget(tier)
    cmd=['/usr/bin/systemd-run','--unit='+unit,'--property=Type=exec','--property=User=embed','--property=WorkingDirectory='+str(ROOT)]
    for p in ['RemainAfterExit=yes','MemoryAccounting=yes','MemoryMax='+str(b['memory_max']),'MemoryHigh='+str(b['memory_high']),'MemorySwapMax=0','LimitMEMLOCK='+str(b['memlock']),'TasksMax=64','CPUQuota=100%','RuntimeMaxSec='+str(b['runtime_seconds']),'TimeoutStopSec=5','KillMode=control-group','Restart=no','NoNewPrivileges=yes','ProtectSystem=strict','ProtectHome=read-only','ReadWritePaths='+str(out)+' '+str(DATA/out.name),'ReadOnlyPaths=/mnt/n0','InaccessiblePaths=-/mnt/n3 -/mnt/n4 -/mnt/n5 -/dev/libnvm0','UMask=0022']:
        cmd.append('--property='+p)
    for k,v in dict(UKL_SOURCE_DIR=str(DATA/out.name),CUDA_VISIBLE_DEVICES=uuid,UKL_MEMORY_TIER_MIB=str(tier),UKL_V6_BOUNDED_WORKER='1',UKL_V6_OUTPUT=str(out),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1').items():cmd.append('--setenv='+k+'='+v)
    return cmd+[PY,'-B','-u','-m','candidates.ukl_memory_ladder_v5.worker']

def show(unit):
    s=run(['/usr/bin/systemctl','show',unit,'-p','LoadState','-p','MainPID','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','MemoryPeak','-p','MemoryMax','-p','MemorySwapMax','-p','LimitMEMLOCK'])
    return dict(x.split('=',1) for x in s.splitlines() if '=' in x)

def exited(state):
    return state.get('ActiveState') in ('inactive','failed') or (state.get('ActiveState')=='active' and state.get('SubState')=='exited' and state.get('MainPID')=='0')

def worker_accepted(state, report, tier):
    size=validate_tier(tier)
    return (state.get('LoadState')=='loaded' and exited(state) and
            state.get('Result')=='success' and state.get('ExecMainStatus')=='0' and
            state.get('MemoryMax')==str(budget(tier)['memory_max']) and state.get('MemorySwapMax')=='0' and
            state.get('LimitMEMLOCK')==str(budget(tier)['memlock']) and
            report.get('passed') is True and report.get('limits_verified_before_cuda') is True and
            report.get('tier_mib')==tier and report.get('loaded_bytes')==size and report.get('gpu_readback_bytes')==size and
            isinstance(report.get('source_sha256'),str) and len(report['source_sha256'])==64 and report['source_sha256']==report.get('gpu_sha256') and
            report.get('normal_unregister') is True and report.get('anonymous_memory_released') is True and report.get('vma_absent_after_release') is True)

def kernel_records(text):
    if len(text)>2*1024**2:raise RuntimeError('Kernel evidence too large; no acceptance')
    records=[json.loads(l) for l in text.splitlines() if l.strip()]
    bad=[r for r in records if BAD.search(str(r.get('MESSAGE',''))) or int(r.get('PRIORITY',6))<=4]
    return records,bad

def probe(base,out,label):
    # Only fsync a fresh 4 KiB test file, never global sync. Parent has bounded wait.
    code="import os,sys; from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(); f=(p/'probe.bin').open('xb'); f.write(bytes(4096)); f.flush(); os.fsync(f.fileno()); f.close(); assert (p/'probe.bin').read_bytes()==bytes(4096)"
    target=base/('ukl_v6_probe_'+out.name+'_'+label)
    log=(out/('probe_'+label+'.log')).open('x');p=subprocess.Popen(['/usr/bin/python3','-B','-c',code,str(target)],stdout=log,stderr=log,start_new_session=True)
    t=time.monotonic()
    try:rc=p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()  # D-state cannot be guaranteed to terminate; do not wait indefinitely.
        return dict(passed=False,timeout=True,pid=p.pid,path=str(target),seconds=time.monotonic()-t)
    finally:log.close()
    return dict(passed=rc==0,exit_code=rc,path=str(target),seconds=time.monotonic()-t)

def write(out,name,d): (out/name).write_text(json.dumps(d,indent=2)+'\n')

def execute(tier):
    global OBSERVER
    b=budget(tier)
    if os.geteuid()!=0:raise RuntimeError('sudo required for systemd limits and complete kernel logs')
    manifest=json.loads((OUT/'manifest.json').read_text())
    for path,h in manifest.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+path)
    # A non-GPU orchestrator retains observation privileges outside the worker cgroup.
    with (ROOT/'results/ukl_runtime_prepare_20261001_v6/smoke.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=prerequisite(tier)
        before=snapshot()
        if not clean_memory(before,before) or before['MemAvailable']<b['host_available_min']:raise RuntimeError('Host admission rejected')
        if shutil.disk_usage('/mnt/n0').free<b['disk_free_min']:raise RuntimeError('Insufficient n0 disk reserve')
        marker=run(['/usr/bin/journalctl','-k','-b','-n','1','--show-cursor','--no-pager'])
        found=re.search(r'^-- cursor: (.+)$',marker,re.M)
        if not found:raise RuntimeError('Cannot establish kernel journal cursor')
        cursor=found[1];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        idx,uuid=gpu();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        out=OUT/('tier_'+str(tier)+'_'+stamp);out.mkdir();account=pwd.getpwnam('embed');os.chown(out,account.pw_uid,account.pw_gid)
        scratch=DATA/out.name;scratch.mkdir(parents=True);os.chown(scratch,account.pw_uid,account.pw_gid)
        unit='digit-ukl-memory5-tier-'+str(tier)+'-'+stamp.replace('_','-')+'.service'
        OBSERVER=Observer((out/'attribution.jsonl').open('x'));OBSERVER.unit=unit;OBSERVER.sample(before)
        write(out,'before.json',dict(tier_mib=tier,budget=b,prerequisite=prior,memory=before,gpu=idx,uuid=uuid,boot_id=boot,kernel_cursor=cursor))
        probes=[probe(ROOT/'results',out,'root_before'),probe(Path('/mnt/n0'),out,'n0_before')]
        write(out,'probes_before.json',probes)
        if not all(x['passed'] for x in probes):raise RuntimeError('Pre-run filesystem probe failed; GPU not launched')
        if gpu()!=(idx,uuid):raise RuntimeError('GPU availability changed; no launch')
        cmd=command(unit,out,uuid,tier);write(out,'command.json',dict(command=cmd))
        error=None;state={};samples=[];launched=False
        try:
            launched=True;run(cmd);deadline=time.monotonic()+b['runtime_seconds']+20
            while time.monotonic()<deadline:
                state=show(unit);s=snapshot();samples.append(s)
                if not clean_memory(before,s) or s['MemAvailable']<64*1024**3:raise RuntimeError('Host memory/dirty admission changed')
                if exited(state):break
                time.sleep(1)
            else:raise RuntimeError('Worker controller deadline exceeded')
        except Exception as e:
            error=repr(e)
            if launched:
                try:run(['/usr/bin/systemctl','stop','--no-block',unit])
                except Exception as stop: error+='; stop='+repr(stop)
        # Postchecks also run for normal nonzero exit. Never launch a second GPU attempt.
        write(out,'memory_samples.json',samples)
        try:
            for _ in range(10):
                state=show(unit)
                if exited(state):break
                time.sleep(1)
            worker_exited=exited(state)
            (out/'worker.log').write_text(run(['/usr/bin/journalctl','-u',unit,'--no-pager','-n','200']))
            post=[snapshot()]
            for _ in range(b['post_observation_seconds']):time.sleep(1);post.append(snapshot())
            journal=run(['/usr/bin/journalctl','-k','-b','--after-cursor='+cursor,'-o','json','--no-pager'])
            (out/'kernel_after.jsonl').write_text(journal);records,bad=kernel_records(journal)
            afterprobes=[]
            if worker_exited:
                afterprobes=[probe(ROOT/'results',out,'root_after'),probe(Path('/mnt/n0'),out,'n0_after')]
            report=json.loads((out/'worker.json').read_text()) if (out/'worker.json').exists() else {}
            passed=not error and worker_exited and worker_accepted(state,report,tier) and not bad and all(clean_memory(before,x) for x in post) and len(afterprobes)==2 and all(x['passed'] for x in afterprobes) and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
            result=dict(tier_mib=tier,manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),passed=bool(passed),error=error,unit=unit,worker_state=state,worker=report,post_memory=post,kernel_records=len(records),kernel_alerts=bad,post_probes=afterprobes,full_graph_enabled=False,raw_ssd_access=False)
        except Exception as e:result=dict(passed=False,error=error,postcheck_error=repr(e),unit=unit,full_graph_enabled=False)
        write(out,'acceptance.json',result);print(json.dumps(dict(output=str(out),**result),indent=2))
        # Evidence is persisted before clearing the retained successful unit.
        if state.get('ActiveState')=='active' and state.get('SubState')=='exited':
            run(['/usr/bin/systemctl','stop',unit])
        OBSERVER.log.close();OBSERVER=None
        if not result['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--tier-gib',type=int,choices=(64,96),default=64);a=p.parse_args();a.tier_mib=a.tier_gib*1024
    if a.execute:execute(a.tier_mib)
    else:print(json.dumps(dict(execute=False,real_gpu_started=False,tier_mib=a.tier_mib,arena_bytes=validate_tier(a.tier_mib),budget=budget(a.tier_mib),worker_runtime_seconds=budget(a.tier_mib)['runtime_seconds'],full_graph_enabled=False,command=command('PLAN.service',OUT/'PLAN','GPU-PLACEHOLDER',a.tier_mib)),indent=2))
