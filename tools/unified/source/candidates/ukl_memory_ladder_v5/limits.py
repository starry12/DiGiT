"""Read effective cgroup v2 and process limits BEFORE CUDA initialization."""
import resource
from .protocol import budget
from pathlib import Path

def check_values(values,tier):
    b=budget(tier)
    expected={'memory.max':b['memory_max'],'memory.swap.max':0,'memory.high':b['memory_high'],'pids.max':64,'memlock_soft':b['memlock'],'memlock_hard':b['memlock']}
    for name,want in expected.items():
        if values.get(name)!=want:raise RuntimeError('Effective limit mismatch: '+name)
    if values.get('cpu_quota',0)<=0 or values['cpu_quota']>values.get('cpu_period',0):raise RuntimeError('CPU quota not bounded to one CPU')

def effective_limits(tier):
    lines=Path('/proc/self/cgroup').read_text().splitlines()
    paths=[x[3:] for x in lines if x.startswith('0::')]
    if len(paths)!=1:raise RuntimeError('Expected unified cgroup v2')
    folder=Path('/sys/fs/cgroup')/paths[0].lstrip('/')
    values={name:int((folder/name).read_text().strip()) for name in ('memory.max','memory.swap.max','memory.high','pids.max')}
    quota,period=(folder/'cpu.max').read_text().split();values.update(cpu_quota=int(quota),cpu_period=int(period))
    soft,hard=resource.getrlimit(resource.RLIMIT_MEMLOCK);values.update(memlock_soft=soft,memlock_hard=hard,cgroup=paths[0])
    check_values(values,tier);return values
