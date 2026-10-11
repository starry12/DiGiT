"""Explicit post-CSC sparse build with a bound hot-set; no implicit cache policy change."""
import argparse,json,subprocess,time
from pathlib import Path
from .transpose import OUT,sha,identity,write

def build(csc,hot,output,replica_cap):
 csc=Path(csc);hot=Path(hot);output=Path(output)
 meta=json.loads((csc/'csc.json').read_text())
 if not meta.get('complete') or meta.get('orientation')!='incoming':raise ValueError('CSC not accepted')
 if hot.stat().st_size!=8*(meta['nodes']//10):raise ValueError('Require the planned 10% logical hot-set, int64 IDs')
 if replica_cap!=(meta['nodes']//5)//2:raise ValueError('This protocol is g2/r20')
 source={str(csc/n):identity(csc/n) for n in ('indptr.i64','indices.i32')};source[str(hot)]=identity(hot)
 binding=dict(source_identity=source,hot_sha256=sha(hot),csc_receipt_sha256=sha(csc/'csc.json'),binary_sha256=sha(OUT/'build/group_overlay'),started=time.time())
 output.mkdir(parents=True,exist_ok=False);write(output/'binding.json',binding)
 try:
  with (output/'build.log').open('w') as log:subprocess.run([str(OUT/'build/group_overlay'),str(csc/'indptr.i64'),str(csc/'indices.i32'),str(hot),str(output),str(replica_cap)],stdout=log,stderr=subprocess.STDOUT,check=True)
  for path,value in source.items():
   if identity(Path(path))!=value:raise RuntimeError('Input changed '+path)
  binding.update(complete=True,seconds=time.time()-binding['started'],files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.suffix in ('.i32','.i64')});write(output/'binding.json',binding)
 except BaseException as exc:
  binding.update(complete=False,error=repr(exc));write(output/'binding.json',binding);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--csc',required=True);p.add_argument('--hot',required=True);p.add_argument('--output',required=True);p.add_argument('--replica-cap',type=int,required=True);a=p.parse_args();build(a.csc,a.hot,a.output,a.replica_cap)
