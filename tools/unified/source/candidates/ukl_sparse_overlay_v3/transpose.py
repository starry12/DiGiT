"""Failure-preserving Java outgoing decoder -> external incoming CSC pipeline."""
import argparse,hashlib,json,os,resource,shutil,subprocess,time
from pathlib import Path
from .protocol import N,E
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/ukl_sparse_overlay_20260930_v3';OLD=ROOT/'results/ukl_window_native_20260930_v2';BIN=OUT/'build/transpose'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
 return h.hexdigest()
def write(p,v):
 t=p.with_suffix('.next');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def identity(p):
 s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)
def compile_java():
 jars=OLD/'tools/jars';lock=json.loads((ROOT/'candidates/ukl_window_native_v2/decoder/dependencies.lock.json').read_text())
 for entry in lock['jars']:
  if sha(jars/entry['file'])!=entry['sha256']:raise RuntimeError('Dependency changed')
 cp=os.pathsep.join(str(jars/x['file']) for x in lock['jars']);classes=OUT/'build/classes';classes.mkdir(exist_ok=True)
 subprocess.run(['java','-XX:ActiveProcessorCount=1','-Xmx512m','-jar',str(jars/'ecj-3.26.0.jar'),'-11','-cp',cp,'-d',str(classes),str(HERE/'decoder/ScatterStream.java')],check=True,timeout=120)
 return str(classes)+os.pathsep+cp

def pipeline(folder,base,n,e,limit,bins=4096,heap='4g',classpath=None,gather=True):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);base=Path(base);cp=classpath or compile_java();sources={str(Path(str(base)+s)):identity(Path(str(base)+s)) for s in ('.graph','.properties')}
 command=['/usr/bin/java','-XX:ActiveProcessorCount=1','-Xmx'+heap,'-cp',cp,'ScatterStream',str(base),str(n),str(e),str(limit)]
 status=dict(stage='starting',pid=os.getpid(),nodes=n,source_edges=e,owner_limit=limit,source_identity=sources,command=command,transpose_binary_sha256=sha(BIN),started=time.time(),complete=False)
 write(folder/'status.json',status)
 decoder=None;consumer=None
 try:
  if gather:
   status['stage']='source_hashing';write(folder/'status.json',status)
   status['source_sha256']={path:sha(path) for path in sources}
   for path,value in sources.items():
    if identity(Path(path))!=value:raise RuntimeError('Source changed while hashing')
   write(folder/'status.json',status)
  with (folder/'decoder.log').open('w') as log,(folder/'scatter.log').open('w') as slog:
   decoder=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
   consumer=subprocess.Popen([str(BIN),'scatter',str(folder),str(bins)],stdin=decoder.stdout,stdout=slog,stderr=subprocess.STDOUT);decoder.stdout.close()
   status.update(stage='scatter',decoder_pid=decoder.pid,consumer_pid=consumer.pid);write(folder/'status.json',status)
   cr=consumer.wait();dr=decoder.wait()
   if cr or dr:raise RuntimeError('scatter/decoder exits %s/%s'%(cr,dr))
  for p,v in sources.items():
   if identity(Path(p))!=v:raise RuntimeError('Source changed '+p)
  scatter=json.loads((folder/'scatter.json').read_text());status.update(stage='scatter_complete',scatter=scatter,scatter_seconds=time.time()-status['started']);write(folder/'status.json',status)
  if gather:
   if limit!=n:raise RuntimeError('Prefix is not a complete graph; cannot publish CSC')
   status['stage']='gather';write(folder/'status.json',status)
   with (folder/'gather.log').open('w') as log:
    subprocess.run([str(BIN),'gather',str(folder),str(n),str(bins),str(e-scatter['removed_self']+n),str(8*2**30)],stdout=log,stderr=subprocess.STDOUT,check=True)
   status['stage']='hashing';write(folder/'status.json',status)
   status['files']={name:dict(bytes=(folder/name).stat().st_size,sha256=sha(folder/name)) for name in ('indptr.i64','indices.i32')}
  status.update(stage='complete' if gather else 'prefix_complete',complete=True,full_csc=gather,seconds=time.time()-status['started'],max_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
  write(folder/'status.json',status);return status
 except BaseException as exc:
  for p in (consumer,decoder):
   if p is not None and p.poll() is None:p.terminate()
  for p in (consumer,decoder):
   if p is not None:
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  status.update(stage='failed',error=repr(exc),complete=False);write(folder/'status.json',status);raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--owners',type=int,default=10000);p.add_argument('--full',action='store_true');args=p.parse_args()
 limit=N if args.full else args.owners
 if not 0<limit<=N:raise ValueError('owner limit')
 out=Path(args.output).resolve()
 if not str(out).startswith('/mnt/n0/digit/ukl_sparse_overlay_v3/'):raise ValueError('Large artifacts must be in independent n0 candidate directory')
 if args.full and shutil.disk_usage(out.parent).free<700*2**30:raise RuntimeError('Need 700 GiB free before full transpose')
 import fcntl
 with open('/tmp/digit-ukl-sparse-build.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  print(json.dumps(pipeline(out,'/mnt/n3/uk2014/uk-2014',N,E,limit,gather=args.full),indent=2),flush=True)
if __name__=='__main__':main()
