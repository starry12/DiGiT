"""Original DiGiT on CPU2 versus default GIDS, with independent evidence."""
import os,sys,json,time,hashlib
from pathlib import Path
from candidates.group_cache_migration_v1.common import stable
stable('IG')
from ae.common import read,write,sha,require,host,append_sync
from submission.v2.common import output_path
from candidates.group_cache_migration_v1.ig.monitor_review import POLICY
from candidates.ig_perf_v5.common import environment,identity,check_inputs,digest,state_hash

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
P=HERE/'protocol.json'
DATA=ROOT/'data/igb'
OUT=ROOT/'results/group_cache_migration_v1/IG'
PARENT_SHA='cec77a5bff14382e89f7ee7fd30660f5802f44265ae74a01b4c70e2d7e123160'

def cfg():return read(P)

def setup():
    os.environ.update(environment())
    sys.path[:0]=[str(ROOT/'candidates/ig_sage_affinity_pair_v1/runtime'),str(ROOT/'ae/igb'),str(ROOT/'ae/igb/runtime')]

def verify():
    from candidates.group_cache_migration_v1.common import verify as shared
    return shared()

def progress(out,stage,**kw):
    value=dict(stage=stage,time_unix=time.time(),pid=os.getpid(),**kw)
    write(Path(out)/'progress.json',value)
    print(json.dumps(value),flush=True)
