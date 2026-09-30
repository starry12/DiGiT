"""CPU-only configuration helpers; no changes to existing experiment inputs."""
import hashlib
import json
import os
import time
from pathlib import Path
from ae.common import sha,read,write,require,host

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
GROUPS=(1,2,4)
PERCENTS=(0,10,20,40,80)

def point(group_size,replica_percent):
    require(type(group_size) is int and group_size in GROUPS,'group_size must be 1, 2 or 4')
    require(type(replica_percent) is int and replica_percent in PERCENTS,'replica_percent must be 0, 10, 20, 40 or 80')
    return dict(id='g%d_r%02d'%(group_size,replica_percent),group_size=group_size,
                replica_percent=replica_percent,replication_ratio=replica_percent/100)

def grid():
    return [point(g,r) for g in GROUPS for r in PERCENTS]

def array(path):
    import numpy as np
    return np.load(path,mmap_mode='r',allow_pickle=False)

def save(path,value):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())

def progress(report,stage,**kw):
    write(Path(report)/'progress.json',dict(stage=stage,updated_unix=time.time(),**kw))

def identity(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def existing_parent(path):
    path=Path(path).resolve()
    while not path.exists():path=path.parent
    require(path.is_dir(),'Expected directory ancestor')
    return path

def code_bindings():
    files=list(HERE.glob('*.py'))
    runtime=ROOT/'candidates/pa_sage_bidir_native_v2/runtime/digit'
    files += [runtime/name for name in ('artifacts.py','io_geometry.py','reorganization.py')]
    files += [ROOT/'candidates/pa_sage_bidir_native_v2/overlay.py',ROOT/'ae/common.py',ROOT/'digit_paths.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}

def budget(n,edges,hot,g,r):
    """Conservative upper bound, not an admission result or measured footprint."""
    require(type(n) is int and n>0 and type(edges) is int and edges>=0,'Invalid graph size')
    require(type(hot) is int and 0<=hot<=n and hot%8==0,'Hot cache must use complete pages')
    point(g,r)
    primary=(n-hot)//g;replica_rows=n*r//100;replicas=replica_rows//g
    groups=primary+replicas
    round8=lambda x:(x+7)//8*8
    rows=groups*8+round8(hot)+round8(n-hot-primary*g)
    payload=rows*512
    # All CSC units remain in the upper bound; actual grouping can reduce it.
    metadata=8*(groups*g+3*groups+rows+n+(n+groups+1)+edges)
    temporary=8*groups*(g+1)+edges+4*n
    overlay=8*(n+groups+1+2*edges)
    peak=payload+metadata+temporary+overlay+max(10*2**30,(payload+metadata+overlay)//10)
    return dict(primary_groups_max=primary,replica_row_budget=replica_rows,
                replica_groups_max=replicas,storage_rows_max=rows,payload_bytes_max=payload,
                metadata_bytes_upper_bound=metadata,overlay_bytes_upper_bound=overlay,filesystem_bytes_with_scratch_and_reserve=peak,
                group_useful_bytes=g*512,group_transfer_bytes=4096,group_padding_bytes=4096-g*512,
                compact_storage_ids_possible=rows-1<=2**31-1,
                gpu_admission='pending_actual_layout_and_live_check',host_admission='pending_live_check')
