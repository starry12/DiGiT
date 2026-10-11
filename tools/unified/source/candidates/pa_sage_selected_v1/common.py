import hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from ae.common import write,sha,require,host,verify_release
CONFIG=HERE/'config.json'
def read(path):return json.loads(Path(path).read_text())
def verify():
 cfg=read(CONFIG);require(verify_release()==cfg['training_release_sha256'],'Base runtime changed')
 for name,digest in read(HERE/'manifest.json')['files'].items():require(sha(HERE/name)==digest,'Frozen file changed: '+name)
 for name,digest in cfg['reference_bindings'].items():require(sha(ROOT/name)==digest,'Provenance reference changed: '+name)
 require(sha(HERE/cfg['checkpoint_file'])==cfg['checkpoint_sha256'],'Frozen checkpoint changed')
 return sha(HERE/'manifest.json')
def identity(path):
 s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def progress(output,stage,**kw):
 value=dict(stage=stage,pid=os.getpid(),time_unix=time.time(),**kw);write(Path(output)/'progress.json',value);print(json.dumps(value),flush=True)
