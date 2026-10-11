"""Bounded nvidia-smi observations; never initialize CUDA or alter another job."""
import subprocess
import time
import ctypes as C
import uuid
from pathlib import Path

def query(args, timings=None):
    begin=time.monotonic();error=None
    try:
        return subprocess.run(['/usr/bin/nvidia-smi',*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5).stdout
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        if timings is not None:
            timings.append(dict(query=list(args),started_monotonic=begin,
                completed_monotonic=time.monotonic(),seconds=time.monotonic()-begin,error=error))

def snapshot(timings=None, selected=None):
    values={}
    scope=[] if selected is None else ['-i',str(selected)]
    for line in query(scope+['--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu'],timings).splitlines():
        f=[v.strip() for v in line.split(',')]
        if len(f)!=5:raise RuntimeError('GPU query format')
        index=int(f[0]);values[index]=dict(index=index,uuid=f[1],used_mib=int(f[2]),free_mib=int(f[3]),util=int(f[4]),pids=[])
    if set(values)!=(set(range(4)) if selected is None else {selected}):raise RuntimeError('GPU index set mismatch')
    by_uuid={v['uuid']:v for v in values.values()}
    for line in query(scope+['--query-compute-apps=gpu_uuid,pid'],timings).splitlines():
        f=[v.strip() for v in line.split(',')]
        if len(f)!=2 or f[0] not in by_uuid:raise RuntimeError('GPU process query format')
        by_uuid[f[0]]['pids'].append(int(f[1]))
    return values

def select(values):
    for i in (2,0,1,3):
        v=values[i]
        if not v['pids'] and v['used_mib']<=256 and v['util']<=5 and v['free_mib']>=24*1024:return i,v['uuid']
    raise RuntimeError('No idle GPU with 24 GiB free; no retry or queue')

def check_idle(selected):
    # Before/after a worker, tolerate a transient query timeout without
    # weakening the successful-query identity, occupancy or memory checks.
    for attempt in range(3):
        try:
            v=snapshot(selected=selected[0])[selected[0]]
            break
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
            if attempt==2:raise
            time.sleep(1)
    if v['uuid']!=selected[1] or v['pids'] or v['used_mib']>256 or v['util']>5 or v['free_mib']<24*1024:raise RuntimeError('Selected GPU is no longer idle')
    return v

from candidates.ukl_training_native_v15r11.gpu_identity import verify_cuda_uuid

from .lean_monitor import GPUConflict

class Monitor:
    def __init__(self,selected,cgroup):self.selected,self.cgroup,self.peak=selected,cgroup,0
    def poll(self):
        timings=[]
        try:v=snapshot(timings,selected=self.selected[0])[self.selected[0]]
        except BaseException as exc:
            exc.query_timings=timings
            raise
        if v['uuid']!=self.selected[1]:raise GPUConflict('GPU identity changed')
        self.peak=max(self.peak,v['used_mib'])
        for pid in v['pids']:
            p=Path('/proc')/str(pid)/'cgroup'
            try:rows=p.read_text().splitlines()
            except FileNotFoundError:continue # Process exited between two read-only observations.
            if not any('0::'+self.cgroup+suffix in rows for suffix in ('','/init','/data')):raise GPUConflict('Foreign process on selected GPU: '+str(pid))
        return dict(**v,peak_used_mib=self.peak,query_timings=timings)
