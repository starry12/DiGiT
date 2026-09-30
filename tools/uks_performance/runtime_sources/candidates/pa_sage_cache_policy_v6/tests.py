"""CPU-only contract regressions; no claims of native acceptance."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .protocol import compile_plan,schedule,validate
from .backend import ExactInstaller,allocation
from .native_adapter import install
from .counters import snapshot,interval,POLICY,GPU,DEVICE
from .common import SCRATCH_BYTES,write
from candidates.pa_sage_cache_policy_v2.tests import FakeStore,counters
from candidates.pa_sage_cache_policy_v5.tests import Tests as WindowTests

class WindowRegression(WindowTests):
    # Retain only the independent counter-math regressions, using v6 functions.
    test_change_scope_preserves_worker_and_training_computation=None
    test_short_origin_hash_mismatch_refused=None
    test_schedule_skips_new_smokes_but_validates_inherited_barrier=None
    def setUp(self):
        self.patcher=patch('candidates.pa_sage_cache_policy_v5.tests.interval',interval)
        self.patcher.start();self.addCleanup(self.patcher.stop)

class Tests(unittest.TestCase):
    def test_budget_schedule_and_protocol_mutations(self):
        p=compile_plan()
        self.assertEqual([x['stage'] for x in schedule(p)],['reuse_preparation']+['smoke']*4+['full']*4+['aggregate'])
        for name,arm in p['arms'].items():
            self.assertEqual(arm['gpu_policy'],'fifo' if name=='digit' else 'legacy')
            self.assertEqual(arm['gpu_feature_cache_bytes'],4*2**30)
            self.assertEqual(arm['cpu_feature_bytes'],5686267904)
            self.assertEqual(arm['lookup_order'],['gpu','cpu','ssd'])
            self.assertEqual(allocation(arm)['gpu_dma_allocation_bytes'],4*2**30)
        bad=copy.deepcopy(p);bad['arms']['degree']['gpu_policy']='bypass'
        with self.assertRaises(ValueError):validate(bad)
    def test_exact_rows_aliases_padding_and_native_modes(self):
        storage=np.array([0,1,-1,2,0,3,2,-1],np.int64)
        hot=np.array([0,2],np.int64);primary=np.array([0,1,3,5],np.int64)
        for name,arm in compile_plan()['arms'].items():
            arm=dict(arm,cpu_rows=2);fs=FakeStore();backend=ExactInstaller(fs)
            receipt=install(backend,arm,hot,primary,storage,chunk_rows=3)
            self.assertEqual(fs.slots,[1,0,0,2,1,0,2,0])
            self.assertEqual(fs.calls[-1],('configure',2 if name=='digit' else 3,4*2**30))
            self.assertEqual(receipt['cpu_feature_bytes'],1024)
            with self.assertRaises(ValueError):backend.configure_gpu_cache('legacy',4*2**30)
    def test_allocator_rejects_missing_budget_or_capability(self):
        from .native_adapter import check_backend
        for policy in ('legacy','fifo'):
            with self.assertRaises(ValueError):allocation(dict(gpu_policy=policy,gpu_feature_cache_bytes=0))
            with self.assertRaises(ValueError):check_backend({},dict(gpu_policy=policy,gpu_feature_cache_bytes=4*2**30))
    def test_loader_switches_only_replacement(self):
        from .runtime import loader_kwargs
        p=compile_plan();p['features']['verified_bytes']=24*512
        m=dict(grouping=dict(group_size=2),feature=dict(row_bytes=512,dim=128,num_storage_rows=24),io=dict(page_size=4096))
        class Geometry:
            def with_payload_offset(self,offset):self.offset=offset;return self
            def to_native_mapping(self):return {'offset':self.offset}
        values=[]
        for name in p['arms']:
            k=loader_kwargs(p,name,m,Geometry())
            self.assertEqual(k.pop('gpu_cache_policy'),'fifo' if name=='digit' else 'legacy')
            self.assertEqual(k['cache_size'],4096);values.append(k)
        self.assertTrue(all(k==values[0] for k in values))
    def test_legacy_counter_accounting_and_wrong_policy_rejection(self):
        a,b=counters(2),counters(2,True)
        for v in (a,b):
            v['policy']['mode']=3;v['gpu']['policy']='legacy';v['gpu']['fifo_ticket']=0
        r=interval(a,b,6,True)
        self.assertEqual(r['policy'],'legacy')
        self.assertEqual(r['serving']['gpu_hit_rows'],1)
        self.assertEqual(r['serving']['cpu_served_rows'],2)
        self.assertEqual(r['serving']['ssd_served_rows'],3)
        b['gpu']['policy']='fifo'
        with self.assertRaises(ValueError):interval(a,b,6,True)
    def test_native_snapshot_checks_actual_allocator(self):
        a=counters(2)
        class Source:
            policy_id=0
            def policy_stats(self):return [dict(a['policy'],mode=3)[k] for k in POLICY]
            def get_feature_access_stats(self):return [0,0]
            def get_gpu_cache_stats(self):return [self.policy_id]+[a['gpu'][k] for k in GPU[1:]]
            def get_device_io_stats(self):return [int(a['device'][k]) for k in DEVICE]
            def get_useful_io_stats(self):return [1,1,0,0,0,0]
        fs=Source();self.assertEqual(snapshot(fs)['gpu']['policy'],'legacy')
        fs.policy_id=1
        with self.assertRaises(ValueError):snapshot(fs)
    def test_old_short_matrix_cannot_unlock_new_run(self):
        from .validation import smoke_matrix
        with self.assertRaises(ValueError):smoke_matrix(dict(schema='digit-cache-inherited-short-matrix-v5'),compile_plan(),'s','p','r')
        from .controller import ensure_short
        class Journal:pass
        with tempfile.TemporaryDirectory() as t:
            j=Journal();j.output=Path(t);p=Path(t)/'protocol.json';q=Path(t)/'prepared.json';write(p,{});write(q,{})
            with self.assertRaises(KeyError):ensure_short(j,compile_plan(),{},p,q,'s')
    def test_comparison_rejects_unequal_gpu_capacity(self):
        from .acceptance import compare,TRACE_KEYS
        reports={}
        for arm in compile_plan()['arms']:
            reports[arm]=dict(cpu_rows=2,hot_nodes_sha256='x',training_seconds=1.,
                region=dict(gpu_feature_cache_bytes=4*2**30,serving={}),**{k:'a'*64 for k in TRACE_KEYS})
        self.assertTrue(compare(reports)['same_total_cache_capacity'])
        reports['degree']['region']['gpu_feature_cache_bytes']=0
        with self.assertRaises(ValueError):compare(reports)

del WindowTests

if __name__=='__main__':unittest.main(verbosity=2)
