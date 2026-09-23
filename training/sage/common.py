import hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ae.common import sha,write,require,host
from ae.pa_sage.common import source_config,splits
P=HERE/'protocol.json'
def read(p):
    from ae.common import read as relocated_read
    return relocated_read(p)
def cfg():return read(P)
def setup():
    from ae.pa_sage.common import setup_imports
    setup_imports()
    sys.path[:0]=[str(HERE/'runtime'),str(ROOT/'runtime/io/runtime')]
def progress(output,stage,**kw):
    v=dict(stage=stage,pid=os.getpid(),time_unix=time.time(),**kw);write(Path(output)/'progress.json',v);print(json.dumps(v),flush=True)
def array_sha(a):
    h=hashlib.sha256()
    for lo in range(0,len(a),1000000):h.update(memoryview(a[lo:lo+1000000]).cast('B'))
    return h.hexdigest()
def verify():
    from artifact_integrity import verify_component
    return verify_component(HERE)
