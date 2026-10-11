"""Read-only host evidence; called outside timed training windows."""
import os
from pathlib import Path
import time

def snapshot():
    mem={k:int(v.split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
         for k,v in [line.split(':',1)] if k in ('MemAvailable','MemFree','SwapFree','Shmem','Dirty')}
    vm=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    proc=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    value=dict(time_unix=time.time(),monotonic=time.monotonic(),memory=mem,
               vm={k:int(vm[k]) for k in ('oom_kill','pswpin','pswpout','pgmajfault') if k in vm},
               process=dict(pid=os.getpid(),minor_faults=int(proc[7]),major_faults=int(proc[9]),
                            user_ticks=int(proc[11]),system_ticks=int(proc[12])))
    for name,path in [('cpu2_frequency_khz','/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq'),
                      ('memory_pressure','/proc/pressure/memory'),('io_pressure','/proc/pressure/io')]:
        try:value[name]=Path(path).read_text().strip()
        except OSError:value[name]=None
    return value
