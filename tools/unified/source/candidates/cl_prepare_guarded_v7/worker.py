"""Bounded CPU CSC -> RevPR -> sparse g2; no GPU and no file-backed registration."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import time
import numpy as np
from . import protocol as P
from .native import Native, direct_hash
from candidates.ukl_native_sampling_v10r4 import fork_guard as H
from candidates.ukl_real_multi_v9.memory import Arena
from candidates.ukl_native_sampling_v10r4.worker import progress_callback
from .start import write


def limits():
    cg=next(l[3:] for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
    path=Path('/sys/fs/cgroup')/cg.lstrip('/')
    v={k:int((path/k).read_text()) for k in ('memory.max','memory.high','memory.swap.max','pids.max')}
    quota,period=map(int,(path/'cpu.max').read_text().split());soft,hard=resource.getrlimit(resource.RLIMIT_MEMLOCK)
    io={l.split()[0]:dict(t.split('=') for t in l.split()[1:]) for l in (path/'io.max').read_text().splitlines()}
    v.update(cgroup=cg,cpu_quota=quota,cpu_period=period,memlock_soft=soft,memlock_hard=hard,io_max=io)
    if not P._limits_valid(v,P.budget(),P.source_device()[1]):raise RuntimeError('Effective limits differ')
    return v


def execute(out,target):
    digest=P.verify_manifest();predecessor=P.require_predecessor();source=P.binding();effective=limits()
    write(out/'effective_limits.json',effective)
    if target.parent!=P.DATA_ROOT or target.is_symlink() or target.stat().st_dev!=source['indices']['identity'][0] or any(target.iterdir()):
        raise RuntimeError('Output scope/filesystem/not-empty')
    def check(remaining=0):
        if (out/'STOP').exists():raise RuntimeError('Controller stop requested')
        available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        if available<remaining+P.RESERVE:raise RuntimeError('Remaining allocation/host reserve')
        if shutil.disk_usage(target).free<32*P.GIB:raise RuntimeError('Disk reserve')
    def interrupted(*a):raise RuntimeError('Termination requested')
    signal.signal(signal.SIGTERM,interrupted)
    last=[0.0];phase=['load_graph']
    def event(stage,**kw):write(out/'status.json',dict(stage=stage,time=time.time(),complete=False,**kw))
    def tick(done=0,total=0,stage=0):
        check(P.HEADROOM)
        if time.monotonic()-last[0]>2:
            event(phase[0],native_stage=stage,done=int(done),total=int(total));last[0]=time.monotonic()
    native=Native();platform.uname() # resolve lazy dependencies before ownership
    ownership=H.OwnershipGuard().enter();arena=hot_arena=None;ptr=idx=hot=part=None;files={}
    try:
        arena=Arena(source,admission_check=lambda remaining:check(remaining+P.HEADROOM),max_bytes=P.EXTENT)
        ownership.track(arena);H.dontfork_arena(arena)
        arena.load(progress_callback(event,lambda:check(arena.remaining_bytes+P.HEADROOM),P.APPLICATION_READ_RATE))
        H.assert_dontfork(arena)
        ptr=np.frombuffer(arena.mm,np.int64,count=P.N+1,offset=arena.offsets['indptr'])
        idx=np.frombuffer(arena.mm,np.int32,count=P.E,offset=arena.offsets['indices'])
        ptr.flags.writeable=idx.flags.writeable=False
        rankdir=target/'rank';rankdir.mkdir();phase[0]='revpr';event(phase[0])
        native.run('rank',ptr,idx,rankdir,callback=tick)
        rank=json.loads((rankdir/'rank.json').read_text())
        for name in ('revpr.f64','hot_nodes.i64'):files[name]=direct_hash(rankdir/name,lambda:check(P.HEADROOM))
        item=files['hot_nodes.i64'];spec=dict(path=item['path'],offset=0,length=item['bytes'],identity=item['identity'],sha256=item['sha256'])
        hot_arena=Arena({'hot':spec},admission_check=lambda remaining:check(remaining+P.HEADROOM),max_bytes=P.GIB)
        ownership.track(hot_arena);H.dontfork_arena(hot_arena)
        hot_arena.load(progress_callback(event,lambda:check(P.HEADROOM),P.APPLICATION_READ_RATE))
        H.assert_dontfork(hot_arena)
        hot=np.frombuffer(hot_arena.mm,np.int64,count=P.N//10);hot.flags.writeable=False
        for lo in range(0,len(hot),1048576):
            check(P.HEADROOM);part=hot[max(0,lo-1):lo+1048576]
            if np.any(part<0) or np.any(part>=P.N) or np.any(part[1:]<=part[:-1]):raise RuntimeError('Hot set order/range')
        part=None
        overlaydir=target/'overlay';overlaydir.mkdir();phase[0]='g2_r20';event(phase[0])
        native.run('group',ptr,idx,overlaydir,hot=hot,callback=tick)
        overlay=json.loads((overlaydir/'overlay.json').read_text());phase[0]='output_readback';event(phase[0])
        for name in ('members.i32','bases.i64','primary.i64','group_ptr.i64','group_ids.i32','covered.i64'):
            files[name]=direct_hash(overlaydir/name,lambda:check(P.HEADROOM))
    finally:
        ptr=idx=hot=part=None;gc.collect()
        for owned in (hot_arena,arena):
            if owned is not None:owned.close();ownership.confirm_released(owned)
        ownership.release()
    if P.binding()!=source or P.verify_manifest()!=digest:raise RuntimeError('Source/code changed')
    report=dict(passed=True,stage='graph',manifest_sha256=digest,predecessor=predecessor,files=files,rank=rank,overlay=overlay,
                effective_limits=effective,gpu_called=False,raw_ssd_access=False,source_revalidated=True,normal_release=True,
                ownership=ownership.receipt(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                application_read_rate=P.APPLICATION_READ_RATE,application_write_rate=P.APPLICATION_WRITE_RATE)
    if not P.validate_worker_report(report):raise RuntimeError('Incomplete CL report')
    write(out/'worker.json',report);write(out/'status.json',dict(complete=True,stage='complete',time=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['graph'],required=True);p.parse_args()
    if os.environ.get('CL_V7_BOUNDED_WORKER')!='1':raise RuntimeError('Bounded controller required')
    execute(Path(os.environ['CL_V7_OUTPUT']).resolve(strict=True),Path(os.environ['CL_V7_DATA']).resolve(strict=True))

if __name__=='__main__':main()
