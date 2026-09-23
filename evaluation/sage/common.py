"""Versioned submission bindings; source and environment paths locate the repository."""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ae.common import read,write,sha,require
from training.sage.common import verify as verify_worker,cfg,setup
BASELINE=ROOT/'reference_results/pa_sage'
POLICY=read(HERE/'monitor_policy.json')

def verify():
    from artifact_integrity import verify_component
    return verify_component(HERE)

def output_path(path):
    p=Path(path).expanduser().resolve()
    require(p!=ROOT/'results' and ROOT/'results' in p.parents,'Use a fresh directory under project results/')
    require(not p.exists(),'Output already exists: '+str(p))
    return p

def environment(gpu=None):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
    env.pop('PYTHONOPTIMIZE',None)
    env['LD_LIBRARY_PATH']=str(ROOT/'third_party/bam/build/lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env

