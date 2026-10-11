import os,sys,json,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=HERE/'protocol.json';DATA=ROOT/'data/igb'
from ae.common import read,write,sha,require,host,append_sync
from submission.v2.common import output_path
from candidates.ig_monitor_v1.review import POLICY
def cfg():return read(P)
def environment(gpu=None):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',USE_DETERMINISTIC_ALG='1',DIGIT_TRAINING_PROFILE='off',DIGIT_VALIDATION_PROFILE='fast',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.pop('PYTHONOPTIMIZE',None)
    env['LD_LIBRARY_PATH']=str(ROOT/'bam/build/lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env
def setup():
    os.environ.update(environment());sys.path[:0]=[str(HERE/'runtime'),str(ROOT/'ae/igb'),str(ROOT/'ae/igb/runtime')]
def verify():
    from submission.v4.common import verify as parent
    m=read(HERE/'manifest.json');require(parent()==m['parent_submission_sha256'],'Parent PA versions changed')
    for name,digest in m['files'].items():require(sha(HERE/name)==digest,'IG candidate changed: '+name)
    for name,digest in m['dependencies'].items():require(sha(ROOT/name)==digest,'IG dependency changed: '+name)
    return sha(HERE/'manifest.json')
def identity(path):
    s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def check_inputs(binding):
    for path,item in binding['files'].items():require(identity(ROOT/path)==item['identity'],'Input changed: '+path)
def progress(out,stage,**kw):
    value=dict(stage=stage,time_unix=time.time(),pid=os.getpid(),**kw);write(Path(out)/'progress.json',value);print(json.dumps(value),flush=True)
def digest(x):
    import numpy as np
    if hasattr(x,'detach'):x=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def state_hash(obj):
    from ig_common import state_hash as original
    return original(obj)
