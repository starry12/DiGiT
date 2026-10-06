import os,resource
from pathlib import Path
from .budget import budget,GIB,MIB
def require_admitted_worker(arm):
    if os.geteuid()!=0 or os.environ.get('UKL_V15_BOUNDED_WORKER')!='1':raise RuntimeError('Fresh admitted GPU/SSD worker required')
    from candidates.ukl_training_native_v15r11.gpu_identity import selection_from_env
    selection_from_env()
    cg=next(l[3:] for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
    from .cgroup_limits import service_group,verify_leaf
    group=service_group(cg)
    p=Path('/sys/fs/cgroup')/group.lstrip('/')
    verify_leaf(Path('/sys/fs/cgroup')/cg.lstrip('/'),p)
    from candidates.cl_training_v11 import protocol as P
    if arm!=P.arm():raise RuntimeError("Stage/arm mismatch")
    from .affinity import verify
    verify(arm)
    b=budget(arm)
    if P.small():
        b.update(host_cgroup_max=8*GIB,host_cgroup_high=7*GIB,memlock=GIB,host_min=72*GIB,gpu_free_min=24*GIB)
    for k,v in [('memory.max',b['host_cgroup_max']),('memory.high',b['host_cgroup_high']),('memory.swap.max',0),('pids.max',64)]:
        if int((p/k).read_text())!=v:raise RuntimeError('Training limit mismatch: '+k)
    q,period=map(int,(p/'cpu.max').read_text().split())
    if q<=0 or q>period or resource.getrlimit(resource.RLIMIT_MEMLOCK)!=(b['memlock'],b['memlock']):raise RuntimeError('Training CPU/MEMLOCK cap')
    from .dataset import source_device
    io={l.split()[0]:dict(x.split('=',1) for x in l.split()[1:]) for l in (p/'io.max').read_text().splitlines()}
    source=io.get(source_device()[1],{})
    if source.get('rbps')!=str(b['source_read_rate']) or source.get('wbps')!=str(b['source_write_rate']):raise RuntimeError('Training source-device I/O cap')
    return b
