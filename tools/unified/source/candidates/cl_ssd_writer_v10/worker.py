"""Fresh process writer; native session is bounded and closed before acceptance."""
import argparse,copy,hashlib,json,os,resource,signal,time
from pathlib import Path
from . import protocol as P
from .engine import run
from .transport import NVMe
from candidates.ukl_native_sampling_v10r4 import fork_guard as H


def limits():
    cg=next(l[3:] for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
    root=Path('/sys/fs/cgroup')/cg.lstrip('/')
    v={k:int((root/k).read_text()) for k in ('memory.max','memory.high','memory.swap.max','pids.max')}
    quota,period=map(int,(root/'cpu.max').read_text().split());soft,hard=resource.getrlimit(resource.RLIMIT_MEMLOCK)
    v.update(cgroup=cg,cpu_quota=quota,cpu_period=period,memlock_soft=soft,memlock_hard=hard,
      io_max={l.split()[0]:dict(x.split('=',1) for x in l.split()[1:]) for l in (root/'io.max').read_text().splitlines()})
    if not P._limits_valid(v,P.budget(),P.source_device()[1]):raise RuntimeError('Writer effective limits mismatch')
    return v


def without_own(snapshot,paths):
    v=copy.deepcopy(snapshot);v.pop('sha256');names={str(p) for p in paths}
    v['files']=[r for r in v['files'] if r['path'] not in names]
    v['regions']=[r for r in v['regions'] if r['label'] not in names]
    for p in paths:
        entries=v['directories'].get(str(p.parent))
        if entries is not None:v['directories'][str(p.parent)]=[n for n in entries if n!=p.name]
    return P.F.digest(v)


def execute(out):
    if os.geteuid()!=0 or os.environ.get('CL_SSD_V10_BOUNDED_WORKER')!='1':raise RuntimeError('Bounded privileged controller required')
    digest=P.verify_manifest();plan,bound,initial=P.inputs(True);effective=limits();P.write(out/'effective_limits.json',effective)
    paths=[P.state_path(a['region']) for a in bound['arms'].values()]
    if any(p.exists() or p.is_symlink() for p in paths):raise RuntimeError('Existing reservations preserved; no automatic resume')
    lease=json.loads((out/'controller_lease.json').read_text())
    states={};last=[0.0];last_inventory=[0.0]
    def check():
        if (out/'STOP').exists():raise RuntimeError('Controller requested stop')
        if Path('/proc/%d/stat'%lease['pid']).read_text().rsplit(')',1)[1].split()[19]!=lease['start_ticks']:raise RuntimeError('Controller lease expired')
        available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        if available<64*P.GIB:raise RuntimeError('Host reserve below 64 GiB')
        if time.monotonic()-last_inventory[0]>10:
            if without_own(P.F.inventory(),paths)!=initial['sha256']:raise RuntimeError('Foreign registration changed')
            last_inventory[0]=time.monotonic()
    def event(**kw):
        if time.monotonic()-last[0]>2:
            P.write(out/'status.json',dict(complete=False,time=time.time(),**kw));last[0]=time.monotonic()
    def record(arm,status,detail):
        states[arm]['status']=status;states[arm]['updated_unix']=time.time();states[arm]['verification']=detail
        P.write(P.state_path(bound['arms'][arm]['region']),states[arm])
    def stop(*args):raise RuntimeError('Termination requested')
    signal.signal(signal.SIGTERM,stop)
    # Reserve guards too. Every status, including failed/unknown, stays occupied.
    for arm,a in bound['arms'].items():
        r=a['region'];state=dict(device='/dev/libnvm0',device_offset_bytes=r['guard_before'][0],
            payload_bytes=r['payload_bytes']+8192,feature_offset_bytes=r['device_offset_bytes'],feature_payload_bytes=r['payload_bytes'],
            source=a['source'],source_binding_sha256=P.F.digest(bound),manifest_sha256=digest,
            arm=arm,status='reserved_authorized_overwrite',automatic_reuse=False,verification={},run=str(out),
            overwrite_authorization_sha256=P.F.digest(P.authorization()))
        P.write(P.state_path(r),state,exclusive=True);states[arm]=state
    guard=H.OwnershipGuard().enter();dev=None;normal=False
    try:
        check();dev=NVMe(P.BINARY,plan['expected_device'],[a['region'] for a in bound['arms'].values()])
        result=run(dev,bound,record,check,event,rate=P.APPLICATION_WRITE_RATE,authorization=P.authorization())
    except BaseException as error:
        for arm in states:record(arm,'failed_occupied',dict(error=repr(error),automatic_reuse=False))
        raise
    finally:
        if dev is not None:dev.close()
        guard.release();normal=True
    if P.verify_manifest()!=digest:raise RuntimeError('Writer code changed')
    P.inputs(False);check()
    result.update(stage='storage',manifest_sha256=digest,source_binding_sha256=P.F.digest(bound),normal_release=normal,
                  gpu_called=False,effective_limits=effective,ownership=guard.receipt(),device_identity_confirmed=True)
    if not P.validate_worker_report(result):raise RuntimeError('Invalid writer receipt')
    P.write(out/'worker.json',result);P.write(out/'status.json',dict(stage='complete',complete=True,time=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['storage'],required=True);p.parse_args()
    execute(Path(os.environ['CL_SSD_V10_OUTPUT']).resolve(strict=True))
if __name__=='__main__':main()
