import json
import os
import time
from .phase import CGROOT,membership,stats


def checkpoint(out,stage):
    row=dict(time=time.time(),stage=stage,pid=os.getpid())
    try:
        group=membership(os.getpid());path=CGROOT/group.lstrip('/')
        row.update(cgroup=group,current=stats(path))
        if path.name in ('init','data'):
            row['phases']={name:stats(path.parent/name) for name in ('init','data')}
    except (OSError,KeyError,ValueError,StopIteration) as error:
        row['diagnostic_error']=repr(error)
    with (out/'file_cache_checkpoints.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    return row
