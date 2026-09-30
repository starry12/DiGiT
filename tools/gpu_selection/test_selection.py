import contextlib,copy,fcntl,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import sys
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'control'))
import gpu_selection as g

def card(i,**kw):return dict(dict(index=i,uuid='GPU-test-%d'%i,name='NVIDIA L40',free_mib=46000,used_mib=0,utilization=0,compute_busy=False),**kw)
class Selection(unittest.TestCase):
 def test_each_card(self):
  for i in range(4):
   with self.subTest(i=i),tempfile.TemporaryDirectory() as d:
    cards=[card(j,compute_busy=j!=i) for j in range(4)]
    with g.reserve(lambda:cards,lambda _:None,Path(d)) as s:self.assertEqual(s['selected']['index'],i)
 def test_no_idle(self):
  with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(RuntimeError,'No idle GPU'):
   with g.reserve(lambda:[card(i,compute_busy=True) for i in range(4)],lambda _:None,Path(d)):pass
 def test_thresholds(self):
  for changes in [dict(free_mib=40959),dict(used_mib=1024),dict(utilization=6),dict(name='Other GPU'),dict(index=4),dict(compute_busy=True)]:
   self.assertFalse(g.eligible(card(0,**changes)))
  self.assertTrue(g.eligible(card(0,used_mib=256),256));self.assertFalse(g.eligible(card(0,used_mib=257),256))
 def test_stability(self):
  self.assertEqual(g.candidates([card(0,compute_busy=True)],[card(0)]),[])
  self.assertEqual(g.candidates([card(0)],[card(0,uuid='GPU-changed')]),[])
 def test_race_fallback(self):
  seq=iter([[card(0),card(1)],[card(0),card(1)],[card(0,compute_busy=True),card(1)],[card(1)]])
  with tempfile.TemporaryDirectory() as d,g.reserve(lambda:next(seq),lambda _:None,Path(d)) as s:self.assertEqual(s['selected']['index'],1)
 def test_lock_held_and_released(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with g.reserve(lambda:[card(0),card(1)],lambda _:None,p) as s:
    with g.reserve(lambda:[card(0),card(1)],lambda _:None,p) as s2:self.assertEqual(s2['selected']['index'],1)
   with g.reserve(lambda:[card(0)],lambda _:None,p) as s:self.assertEqual(s['selected']['index'],0)
 def test_symlink_lock_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'target').touch();(p/'uks-gpu-GPU-test-0.lock').symlink_to(p/'target')
   with self.assertRaises(OSError):
    with g.reserve(lambda:[card(0)],lambda _:None,p):pass
 def test_assignment_propagation(self):
  with patch.dict(os.environ,{},clear=True):
   for i in range(4):
    g.install_assignment(dict(selected={k:card(i)[k] for k in ('index','uuid','name')}))
    self.assertEqual(g.propagated()['CUDA_VISIBLE_DEVICES'],str(i));self.assertEqual(g.assignment()['index'],i)
   os.environ['CUDA_VISIBLE_DEVICES']='0'
   with self.assertRaises(RuntimeError):g.assignment()
 def test_idle_uuid_rejected(self):
  with patch.dict(os.environ,{},clear=True):
   g.install_assignment(dict(selected={k:card(0)[k] for k in ('index','uuid','name')}))
   with patch.object(g,'query',return_value=[card(0,uuid='GPU-changed')]),self.assertRaises(RuntimeError):g.idle()
 def test_query(self):
  with patch.object(g.subprocess,'check_output',side_effect=['0, GPU-test-0, NVIDIA L40, 46000, 0, 20\n','GPU-test-0, 12\n']):
   c=g.query()[0];self.assertTrue(c['compute_busy']);self.assertEqual(c['used_mib'],20)
 def test_query_failure_closed(self):
  with patch.object(g.subprocess,'check_output',side_effect=['0, GPU-test-0, NVIDIA L40, 46000, 0, 20\n','unreadable\n']),self.assertRaises(RuntimeError):g.query()
 def test_monitor_gpu(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True):
   g.install_assignment(dict(selected={k:card(1)[k] for k in ('index','uuid','name')}));p=Path(d)/'summary.json'
   p.write_text(json.dumps(dict(gpu='1',physical_gpu_uuid='GPU-test-1')));g.verify_monitor(d)
   p.write_text(json.dumps(dict(gpu='2')))
   with self.assertRaises(RuntimeError):g.verify_monitor(d)
 def test_transport_integrity(self):self.assertEqual(len(g.transport()),64)
 def test_source_syntax(self):
  for p in [B.parents[1]/v['source'] for v in json.loads((B/'deployment.json').read_text())['files'] if v['source'].endswith('.py')]:compile(p.read_text(),str(p),'exec')
if __name__=='__main__':unittest.main(verbosity=2)
