import copy,subprocess,unittest
from types import SimpleNamespace
from unittest.mock import patch
from .common import PY,ROOT
from .affinity import apply_affinity,validate_affinity,install
from .tests import fixture
from .report import summarize,validate_worker


class Affinity(unittest.TestCase):
    def test_pin_every_existing_thread(self):
        before=dict(thread_masks={'101':[0,1,2,3],'102':[0,1,2,3]})
        after=dict(thread_masks={'101':[2],'102':[2]})
        for arm in ('gids','digit_full'):
            with patch('candidates.uks_group_first_cpu2_v1.affinity.snapshot',side_effect=[before,after]), \
                 patch('candidates.uks_group_first_cpu2_v1.affinity.os.sched_getaffinity',return_value={0,1,2,3}), \
                 patch('candidates.uks_group_first_cpu2_v1.affinity.os.sched_setaffinity') as setter:
                value=apply_affinity(arm)
                self.assertEqual([c.args for c in setter.call_args_list],[(101,{2}),(102,{2})])
                self.assertEqual(value['arm'],arm);self.assertEqual(value['policy'],'one_core')

    def test_reject_unbound_or_escaped_thread(self):
        for mode in ('missing','policy','cpu','initial','final'):
            reports=fixture();r=reports['round2_gids']
            if mode=='missing':r['matched_cpu_affinity']=False
            if mode=='policy':r['affinity']['initial']['policy']='unchanged'
            if mode=='cpu':r['affinity']['initial']['cpu']=3
            if mode=='initial':r['affinity']['initial']['after']['thread_masks']['9']=[0,2]
            if mode=='final':r['affinity']['final']['thread_masks']['9']=[0,2]
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):summarize(reports)
            with self.assertRaises(RuntimeError):validate_worker(r,'performance','gids','source','prepared',2)

    def test_process_local_dispatch_restored_after_failure(self):
        from candidates.ig_sage_affinity_pair_v1 import affinity as original
        old=(original.apply_affinity,original.validate_affinity)
        with patch('candidates.uks_group_first_cpu2_v1.affinity.apply_affinity',return_value={'mock':True}):
            with self.assertRaises(ValueError):
                with install() as values:
                    self.assertEqual(original.apply_affinity('gids'),{'mock':True})
                    self.assertEqual(values,[{'mock':True}]);raise ValueError('injected')
        self.assertEqual(old,(original.apply_affinity,original.validate_affinity))

    def test_real_child_and_new_thread_inherit_cpu2(self):
        code='''import threading
from candidates.uks_group_first_cpu2_v1.affinity import apply_affinity,validate_affinity,snapshot
value=apply_affinity('gids')
started=threading.Event();finish=threading.Event()
def child():
    started.set();finish.wait()
t=threading.Thread(target=child);t.start();started.wait()
try:validate_affinity(dict(initial=value,final=snapshot()),'gids')
finally:finish.set();t.join()
'''
        subprocess.run([PY,'-B','-c',code],cwd=ROOT,check=True,timeout=30)

    def test_wait_for_whole_pa_or_uks_chain(self):
        from .resources import prior_controller_busy
        text='digit-pa-tiered-nowb20-v1-test.service loaded active running PA\n'
        text+='digit-uks-group-first-compare-test.service loaded deactivating stop UKS\n'
        text+='digit-uks-old.service loaded failed failed Old\n'
        with patch('candidates.uks_group_first_cpu2_v1.resources.subprocess.run',return_value=SimpleNamespace(stdout=text)):
            busy=prior_controller_busy();self.assertEqual(len(busy),2)
            self.assertIn('digit-pa-tiered-nowb20',busy[0])
