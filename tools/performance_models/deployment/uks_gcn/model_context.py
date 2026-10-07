"""Fixed UKS/gcn protocol; isolated service, no user-selected Python code."""
import hashlib,json,sys
from pathlib import Path
BASE=Path('/srv/digit-ae/admin/multimodel_v2')
sys.path.insert(0,str(BASE))
sys.path.insert(0,'/home/embed/digit')
from runtime.models import identity,require_identity,cpu_check
DATASET='UKS'
MODEL='gcn'
EXPECTED_RUNTIME_SHA='f615fd751b4c24b2993aa9f383f5ef1da9bf87cfa9c54059d6318c2ae6d03dd0'
EXPECTED=identity(DATASET,MODEL,EXPECTED_RUNTIME_SHA)
def verify():
 manifest=BASE/'runtime_manifest.json'
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=EXPECTED_RUNTIME_SHA:raise RuntimeError('Model adapter manifest changed')
 for name,digest in json.loads(manifest.read_text())['files'].items():
  p=BASE/'runtime'/name
  if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('Model adapter changed: '+name)
 control=Path(__file__).resolve().parent
 transport=control/'transport_manifest.json'
 for name,digest in json.loads(transport.read_text())['files'].items():
  p=control/name
  if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('Model service changed: '+name)
 EXPECTED['service_sha256']=hashlib.sha256(transport.read_bytes()).hexdigest()
 return EXPECTED
def activate():
 verify()
 # Install UUID-compatible identity hooks before importing any parent module
 # that captures an identity function with a from-import.
 sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
 import stable_identity
 stable_identity.install(DATASET)
 from runtime.adapter import activate as apply
 apply(DATASET,MODEL,EXPECTED)
def checked(report):
 require_identity(report,verify())
 if DATASET in ('IG','UKL','CL'):
  from runtime.gates import validate as gate_valid
  gates=[row.get('warmup_gate') for row in report['rows']] if DATASET=='IG' and 'rows' in report else [report.get('warmup_gate')]
  if not gates or not all(gate_valid(g) for g in gates):raise RuntimeError('Missing new-model native warmup gate')
 if DATASET in ('UKL','CL') and MODEL=='gat':
  from runtime.probe import validate
  if not validate(report.get('model_memory_probe',{}),DATASET,MODEL):raise RuntimeError('Missing bounded GAT memory probe')
 return report
def selftest():
 result=cpu_check(DATASET,MODEL)
 print(json.dumps(dict(result,model_identity=verify())))
 return result
