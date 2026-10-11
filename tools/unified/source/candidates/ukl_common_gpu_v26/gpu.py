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
        if not v['pids'] and v['used_mib']<=256 and v['util']<=5 and v['free_mib']>=44*1024:return i,v['uuid']
    raise RuntimeError('No idle GPU with 44 GiB free; no retry or queue')

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
    if v['uuid']!=selected[1] or v['pids'] or v['used_mib']>256 or v['util']>5 or v['free_mib']<44*1024:raise RuntimeError('Selected GPU is no longer idle: '+repr(v))
    return v

def wait_released(selected, record=lambda v:None, timeout=30):
    """Post-exit only: retain thresholds and require two idle observations.

    A process or identity change aborts immediately. Only process-free memory
    cleanup/utilization decay and bounded query errors may settle.
    """
    started=time.monotonic();deadline=started+timeout;consecutive=0;observations=[]
    while time.monotonic()<deadline:
        timings=[]
        try:
            v=snapshot(timings=timings,selected=selected[0])[selected[0]]
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
            consecutive=0
            entry=dict(seconds=time.monotonic()-started,error=repr(exc),query_timings=timings)
        else:
            entry=dict(seconds=time.monotonic()-started,snapshot=v,query_timings=timings)
        observations.append(entry);record(entry)
        if 'snapshot' in entry:
            if v['uuid']!=selected[1] or v['pids']:
                raise RuntimeError('GPU identity/process conflict during release: '+repr(v))
            idle=v['used_mib']<=256 and v['free_mib']>=44*1024 and v['util']<=5
            consecutive=consecutive+1 if idle else 0
            if consecutive>=2 and time.monotonic()<=deadline:
                return dict(passed=True,seconds=time.monotonic()-started,observations=observations)
        remaining=deadline-time.monotonic()
        if remaining>0:time.sleep(min(1,remaining))
    raise RuntimeError('GPU release did not settle within %s seconds; last observation: %r' % (timeout,observations[-1:] ))

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


def kernel_identity(selected):
    rows=query(['-i',str(selected[0]),'--query-gpu=index,uuid,pci.bus_id']).strip().splitlines()
    if len(rows)!=1:raise RuntimeError('Ambiguous selected GPU identity')
    fields=[x.strip() for x in rows[0].split(',')]
    if len(fields)!=3 or int(fields[0])!=selected[0] or fields[1]!=selected[1]:raise RuntimeError('Selected GPU identity changed')
    from .kernel_scope import pci_key
    pci_key(fields[2])
    return dict(index=selected[0],uuid=selected[1],pci=fields[2])
