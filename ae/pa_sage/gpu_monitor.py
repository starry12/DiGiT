"""Small standalone GPU sampler; deliberately imports no project/CUDA modules."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def monitor_memory():
    fields={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key,value,*_=line.split();fields[key.rstrip(':')]=int(value)*1024
    return fields


def sample(gpu):
    began=time.time()
    value=subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
    fields=value.strip().split(',')
    if len(fields)!=2:raise ValueError('Expected one GPU memory/utilization row')
    memory,utilization=[int(x.strip()) for x in fields]
    if memory<0 or not 0<=utilization<=100:raise ValueError('Invalid GPU sample')
    return dict(time_unix=time.time(),query_started_unix=began,device_used_bytes=memory*2**20,
        utilization_percent=utilization,monitor_rss_bytes=monitor_memory()['VmRSS'],monitor_pid=os.getpid(),monitor_parent_pid=os.getppid())


def main(output,gpu,period):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    stop=threading.Event()
    def stopping(signum,frame):stop.set()
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    state=dict(mode='external_small_process',pid=os.getpid(),parent_pid=os.getppid(),gpu=str(gpu),
        started_unix=time.time(),samples=0,errors=[],peak_device_used_bytes=0,period_after_query_seconds=period)
    try:
        with (output/'samples.jsonl').open('x',buffering=1) as stream:
            while not stop.is_set():
                try:
                    rec=sample(gpu)
                    state['samples']+=1
                    state['peak_device_used_bytes']=max(state['peak_device_used_bytes'],rec['device_used_bytes'])
                    stream.write(json.dumps(rec)+'\n')
                    if state['samples']==1:write(output/'ready.json',dict(pid=os.getpid(),passed=True,first_sample=rec))
                except Exception as exc:
                    rec=dict(time_unix=time.time(),error=type(exc).__name__+': '+str(exc))
                    state['errors'].append(rec);stream.write(json.dumps(rec)+'\n')
                    if state['samples']==0:raise
                stop.wait(period)
    finally:
        state.update(finished_unix=time.time(),peak_rss_bytes=monitor_memory()['VmHWM'],
            getrusage_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            memory_note='VmHWM measures this exec address space; getrusage high-water can retain pre-exec parent history.',
            complete=True,passed=state['samples']>0 and not state['errors'])
        write(output/'summary.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',default='2');p.add_argument('--period',type=float,default=.5)
    a=p.parse_args()
    if a.period<=0:p.error('period must be positive')
    main(a.output,a.gpu,a.period)
