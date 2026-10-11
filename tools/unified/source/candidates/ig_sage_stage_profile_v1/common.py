"""Symmetric host instrumentation over the accepted IG 300-batch implementation."""
import os,sys,json,time,hashlib
from pathlib import Path
from ae.common import read,write,sha,require,host,append_sync
from submission.v2.common import output_path
from candidates.ig_monitor_v1.review import POLICY
from candidates.ig_perf_v5.common import environment,identity,check_inputs,digest,state_hash

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
P=HERE/'protocol.json'
DATA=ROOT/'data/igb'
OUT=ROOT/'results/ig_sage_stage_profile_20260926_v1'
PARENT_SHA='cab778971d77b7cb8001e3311b2f2e1ffb3ba343b0652e4ad896eadc1351841f'

def cfg():return read(P)

def setup():
    os.environ.update(environment())
    sys.path[:0]=[str(HERE/'runtime'),str(ROOT/'ae/igb'),str(ROOT/'ae/igb/runtime')]

def verify():
    from candidates.ig_perf_window300_v1.common import verify as parent
    require(parent()==PARENT_SHA,'Accepted IG parent changed')
    manifest=read(HERE/'manifest.json')
    require(manifest['parent_candidate_sha256']==PARENT_SHA,'Wrong parent')
    for name,expected in manifest['files'].items():
        require(sha(HERE/name)==expected,'Stage profile source changed: '+name)
    require(sha(OUT/'cpu_checks.json')==manifest['cpu_checks_sha256'],'Preparation tests changed')
    require(sha(HERE/'runtime/IGPerfNative.so')==sha(ROOT/'candidates/ig_perf_v5/runtime/IGPerfNative.so'),
            'Native implementation differs from accepted IG')
    return sha(HERE/'manifest.json')

def progress(out,stage,**kw):
    value=dict(stage=stage,time_unix=time.time(),pid=os.getpid(),**kw)
    write(Path(out)/'progress.json',value)
    print(json.dumps(value),flush=True)
