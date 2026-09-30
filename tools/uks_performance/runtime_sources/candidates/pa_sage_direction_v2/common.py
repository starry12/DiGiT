import hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ae.common import read,write,sha,require,host
PROTOCOL=HERE/'protocol.json'
def verify():
 from ae.common import verify_release
 require(verify_release()==read(PROTOCOL)['training_release_sha256'],'Base training release changed')
 for rel,digest in read(PROTOCOL)['reference_bindings'].items():require(sha(ROOT/rel)==digest,'Reference changed: '+rel)
 m=read(HERE/'manifest.json')
 for name,digest in m['files'].items():require(sha(HERE/name)==digest,'Candidate changed: '+name)
 return sha(HERE/'manifest.json')
def identity(p):
 s=Path(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def progress(output,stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),time_unix=time.time(),**kw);write(Path(output)/'progress.json',v);print(json.dumps(v),flush=True)
