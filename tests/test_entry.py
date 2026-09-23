"""Boundary checks for the single submission entry, with no device access."""
import argparse,contextlib,io,json,os,sys,tempfile,unittest
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import artifact
class EntryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):(ROOT/'results').mkdir(exist_ok=True)
 def test_direct_model_dispatch(self):
  for model,prefix in [('sage','evaluation.sage'),('gcn','training.gcn'),('gat','training.gat')]:
   for action,suffix in [('check','check'),('smoke','controller'),('representative','controller'),('summarize','summarize')]:
    a=argparse.Namespace(action=action,model=model,gpu=2)
    cmd=artifact._command(a,ROOT/'results/new',ROOT/'results/prior')
    self.assertIn(prefix+'.'+suffix,cmd)
    self.assertFalse(any('submission/v' in x or 'candidates.' in x for x in cmd))
 def test_reject_unsupported_dataset(self):
  with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):artifact.main(['matrix','--dataset','IG'])
 def test_output_boundary(self):
  for p in ['/tmp/escape','results/../escape','results',str(ROOT/'training')]:
   with self.assertRaises((ValueError,RuntimeError)):artifact._result_path(p)
 def test_output_symlink_rejected(self):
  (ROOT/'results').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as d,tempfile.TemporaryDirectory() as ext:
   p=Path(d)/'link';p.symlink_to(ext,target_is_directory=True)
   with self.assertRaises((ValueError,RuntimeError)):artifact._result_path(p/'new')
 def test_old_or_failed_results_rejected(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as d:
   p=Path(d)
   with self.assertRaises(RuntimeError):artifact._new_result(p,'expected','PA','sage')
   (p/'artifact_invocation.json').write_text(json.dumps({'schema':'digit-ae-artifact-invocation-v1','package_sha256':'expected','action':'representative','exit_code':1,'dataset':'PA','model':'sage','output':str(p.relative_to(ROOT))}))
   with self.assertRaises(RuntimeError):artifact._new_result(p,'expected','PA','sage')
 def test_incomplete_preflight_rejected(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as d:
   p=Path(d);(p/'status.json').write_text('{"passed":true,"complete":false}')
   with self.assertRaises(RuntimeError):artifact._check_status(p)
 def test_receipt_protocol_bytes_preserved(self):
  import hashlib
  from digit_paths import json_loads
  for model in ['gcn','gat']:
   p=json.loads((ROOT/'training'/model/'protocol.json').read_text())
   self.assertEqual(p['data_preparation_protocol_sha256'],hashlib.sha256((ROOT/'training/sage/protocol.json').read_bytes()).hexdigest())
  parsed=json_loads((ROOT/'training/sage/protocol.json').read_text())
  self.assertEqual(parsed['selected_config'],'configs/sage_selected.json')
 def test_counter_corruption_rejected(self):
  from runtime.io.accounting import decode,useful_interval
  a=decode([1,1,0,0,0,0]);b=decode([1,1,4096,512,512,1])
  v=useful_interval(a,b,{'gpu_ssd':1});self.assertEqual(v['ssd_useful_bytes'],512)
  with self.assertRaises(RuntimeError):useful_interval(a,decode([1,2,4096,512,512,1]),{'gpu_ssd':1})
  with self.assertRaises(RuntimeError):useful_interval(a,decode([1,1,4096,1024,512,1]),{'gpu_ssd':1})
if __name__=='__main__':unittest.main(verbosity=2)
