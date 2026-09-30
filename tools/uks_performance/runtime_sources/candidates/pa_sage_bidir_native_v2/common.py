import hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ae.common import sha,write,require,host
from ae.pa_sage.common import source_config,splits
P=HERE/'protocol.json'
def read(p):return json.loads(Path(p).read_text())
def cfg():return read(P)
def setup():
    from ae.pa_sage.common import setup_imports
    setup_imports()
    sys.path[:0]=[str(HERE/'runtime'),str(ROOT/'candidates/io_accounting_v1/runtime')]
def progress(output,stage,**kw):
    v=dict(stage=stage,pid=os.getpid(),time_unix=time.time(),**kw);write(Path(output)/'progress.json',v);print(json.dumps(v),flush=True)
def array_sha(a):
    h=hashlib.sha256()
    for lo in range(0,len(a),1000000):h.update(memoryview(a[lo:lo+1000000]).cast('B'))
    return h.hexdigest()
def verify():
    from candidates.io_accounting_v1.common import verify_release
    m=read(HERE/'manifest.json');require(verify_release()==m['io_candidate_sha256'],'I/O candidate changed')
    from candidates.pa_sage_bidir_native_v1.common import verify as verify_parent
    require(verify_parent()==m['parent_candidate_sha256'],'Parent candidate changed')
    for rel,digest in m['files'].items():require(sha(HERE/rel)==digest,'Candidate changed: '+rel)
    for rel,digest in m['external_files'].items():require(sha(ROOT/rel)==digest,'Dependency changed: '+rel)
    return sha(HERE/'manifest.json')
