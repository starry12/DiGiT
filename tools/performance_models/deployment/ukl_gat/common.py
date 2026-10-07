"""Fixed paths and authenticated immutable AE runtime inputs."""
import model_context
import hashlib,json,os
from pathlib import Path
ROOT=Path('/home/embed/digit')
CONTROL=Path('/srv/digit-ae/admin/multimodel_v2/ukl_gat')
SNAPSHOT=Path('/srv/digit-ae/releases/ukl_sage_performance_v27_20261006')
OUTPUT=Path('/srv/digit-ae/ukl-gat-performance-results')
PROBES=Path('/mnt/n0/digit/ae_ukl_probes')
UNIT='digit-ae-ukl-gat-performance.service'
PYTHON='/srv/digit-ae/env/bin/python'
LOCK=ROOT/'results/ukl_runtime_prepare_20261001_v6/smoke.lock'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def request_path(value):
 p=Path(value);require(p.parent==OUTPUT and p.resolve()==p and not p.is_symlink(),'Invalid request directory');return p
def transport_identity():
 m=read(CONTROL/'transport_manifest.json')
 for n,d in m['files'].items():
  p=CONTROL/n;require(not p.is_symlink() and sha(p)==d,'Transport changed: '+n)
 return sha(CONTROL/'transport_manifest.json')
def identity():
 require(sha(ROOT/'ae_ukl_snapshot.json')==sha(CONTROL/'snapshot_identity.json'),'Wrong runtime namespace')
 m=read(CONTROL/'snapshot_manifest.json')
 for n,d in m['files'].items():
  p=ROOT/n;require(not p.is_symlink() and sha(p)==d,'Snapshot changed: '+n)
 return dict(model_identity=model_context.verify(),snapshot_sha256=sha(CONTROL/'snapshot_manifest.json'),transport_sha256=transport_identity(),
             source_sha256=m['source_manifest_sha256'])
def configure(experiment):
 from candidates.ukl_common_gpu_five_v27 import protocol as P
 P.OUT=Path(experiment);P.PYTHON=PYTHON;P.LOCK=CONTROL/'state/controller.lock'
 return P
