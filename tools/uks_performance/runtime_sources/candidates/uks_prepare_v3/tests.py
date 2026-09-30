"""Bounded CPU regressions; CUDA compilation and native execution are deferred."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from .common import ROOT,HERE,OUT,GRID,ARMS,require,read,sha,write_new,cpu_gate,require_grid_complete
from .protocol import compile_plan,validate,fixture_plan


class Tests(unittest.TestCase):
    def test_complete_epoch_protocol_and_reference_choices(self):
        p=validate(compile_plan());m=p['measurement']
        self.assertEqual(m['updates'],13051);self.assertEqual(m['last_batch'],104)
        self.assertEqual(m['training_examples'],13363304);self.assertEqual(m['warmup_batches'],0)
        self.assertEqual(p['feature_dim'],256);self.assertEqual(p['features']['row_bytes'],1024)
        self.assertEqual(p['cpu_cache_rows']*1024,13684023296)
        self.assertFalse(p['features']['shared_proxy']['approved_for_uks'])
        self.assertIsNone(p['host_required_bytes']);self.assertIsNone(p['ssd_offsets'])
        self.assertEqual(p['graph_contract']['sampling_mode'],'gpu_i32_uva_eid64')
        self.assertFalse(p['execution']['native_ready'])

    def test_protocol_rejects_legacy_windows_width_or_unvalidated_proxy(self):
        for path,value in [(('measurement','warmup_batches'),20),(('features','row_bytes'),512),
                           (('features','mode'),'physical_row_proxy'),(('execution','native_ready'),True)]:
            bad=compile_plan();bad[path[0]][path[1]]=value
            with self.assertRaises(ValueError):validate(bad)

    def test_grid_gate_requires_exact_fifteen_passed_points(self):
        valid=dict(complete=True,passed=True,completed=['g%d_r%02d'%(g,r) for g in (1,2,4) for r in (0,10,20,40,80)])
        require_grid_complete(valid)
        for change in (dict(complete=False),dict(passed=False),dict(completed=valid['completed'][:-1]),
                       dict(completed=['other'+str(i) for i in range(15)])):
            with self.assertRaises(ValueError):require_grid_complete(dict(valid,**change))

    def test_execution_gate_precedes_mutations_and_child_launch(self):
        from .cli import main
        with patch('sys.argv',['cli','run','--execute','--output','/tmp/uks_must_not_create']), \
             patch('candidates.uks_prepare_v3.common.read',return_value=dict(complete=False,passed=False)), \
             patch('pathlib.Path.mkdir') as mkdir,patch('subprocess.run') as child:
            with self.assertRaises(ValueError):main()
            mkdir.assert_not_called();child.assert_not_called()

    def test_heavy_work_not_unblocked_by_legacy_ig_completion(self):
        from .common import heavy_gate
        with patch('candidates.uks_prepare_v3.common.read',return_value=dict(stage='layout_g1_r10',complete=False)), \
             patch('candidates.uks_sage_v2.safety.active_ig',return_value=dict(complete=True)):
            with self.assertRaises(ValueError):heavy_gate()

    def test_build_and_native_import_rejected_before_compiler_or_cuda(self):
        from . import build,runtime
        with patch('candidates.uks_prepare_v3.build.read',return_value=dict(complete=False)), \
             patch('subprocess.run') as child,patch('pathlib.Path.mkdir') as mkdir:
            with self.assertRaises(ValueError):build.execute()
            child.assert_not_called();mkdir.assert_not_called()
        with patch('candidates.uks_prepare_v3.common.read',return_value=dict(complete=False)), \
             patch('importlib.import_module') as imp:
            with self.assertRaises(ValueError):runtime.prepare_imports()
            imp.assert_not_called()

    def test_cpu_checks_defer_before_native_measurements(self):
        for stage in ('native_g2_r20','budget_g1_r10','waiting_native_g1_r10','unknown'):
            with patch('candidates.uks_prepare_v3.common.read',return_value=dict(stage=stage,complete=False)):
                with self.assertRaises(ValueError):cpu_gate()

    def test_full_root_order_is_reproducible_and_covers_tail(self):
        from .data import fixture_inputs,epoch_order,batches
        p=fixture_plan();f,y,selected=fixture_inputs(p);a,r=epoch_order(p,selected);b,s=epoch_order(p,selected)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(np.sort(a),selected)
        self.assertEqual(r,s);self.assertEqual([len(x) for x in batches(a,8)],[8,8,8,1])
        self.assertEqual(len(np.unique(a)),25);self.assertEqual(y.dtype,np.int64)
        self.assertEqual(f.shape,(257,256));self.assertTrue(np.isfinite(f).all())

    def test_synthetic_seeds_independent_and_fixed_block_prefix(self):
        from .data import row_block
        p=fixture_plan();a=row_block(p,0,257);short=row_block(p,0,5);np.testing.assert_array_equal(a[:5],short)
        labels=row_block(p,0,257,True);p['synthetic']['feature_seed']+=1
        self.assertFalse(np.array_equal(a,row_block(p,0,257)))
        np.testing.assert_array_equal(labels,row_block(p,0,257,True))
        self.assertTrue(((labels>=0)&(labels<19)).all())

    def test_large_split_generation_is_gated(self):
        from .data import training_ids,row_block
        with patch('candidates.uks_prepare_v3.common.read',return_value=dict(complete=False)),patch('numpy.random.Generator') as rng:
            for fn in (lambda:training_ids(compile_plan()),lambda:row_block(compile_plan(),0,10)):
                with self.assertRaises(ValueError):fn()
            rng.assert_not_called()

    def test_split_rejects_duplicate_or_out_of_range_ids(self):
        from .data import epoch_order
        p=fixture_plan();ids=np.arange(25,dtype=np.int64);ids[-1]=ids[-2]
        with self.assertRaises(ValueError):epoch_order(p,ids)
        ids[-1]=257
        with self.assertRaises(ValueError):epoch_order(p,ids)

    def test_epoch_ledger_accepts_13051_updates_and_rejects_short_window(self):
        from .epoch import Coverage
        c=Coverage(13363304,1024)
        for _ in range(13050):c.observe(1024)
        with self.assertRaises(ValueError):c.finish()
        with self.assertRaises(ValueError):c.observe(1024)
        c.observe(104);r=c.finish();self.assertEqual(r['updates'],13051)
        with self.assertRaises(ValueError):c.observe(104)
        old=Coverage(13363304,1024)
        for _ in range(110):old.observe(1024)
        with self.assertRaises(ValueError):old.finish()

    def test_64bit_offsets_and_eids_above_uint32_without_huge_arrays(self):
        from .graph import offsets64,edge_ids64,compact_ids
        total=5507679822;ptr=np.array([0,2**32+17,total],np.int64);ids=np.array([2**32,2**32+17,total-1],np.int64)
        np.testing.assert_array_equal(offsets64(ptr,total),ptr);np.testing.assert_array_equal(edge_ids64(ids,total),ids)
        for values in (ids.astype(np.uint32),ids.astype(np.float64)):
            with self.assertRaises(ValueError):edge_ids64(values,total)
        with self.assertRaises(ValueError):offsets64(ptr.astype(np.uint32),total)
        with self.assertRaises(ValueError):compact_ids(ids)
        self.assertEqual(compact_ids(np.array([-1,0,133633039],np.int64)).dtype,np.int32)

    def test_csc_monotonicity_and_eid_bounds(self):
        from .graph import offsets64,edge_ids64
        for a in ([0,10,9,20],[1,10,20],[0,10,21]):
            with self.assertRaises(ValueError):offsets64(np.array(a,np.int64),20)
        for a in ([-1],[20]):
            with self.assertRaises(ValueError):edge_ids64(np.array(a,np.int64),20)

    def test_graph_normalization_preserves_directed_multiedges_and_single_selfloops(self):
        from .graph import normalized_fixture
        edge=np.array([[2,2,1,1,0],[1,1,1,1,2]],np.int64)
        ptr,idx,eids=normalized_fixture(edge,4)
        actual=[(int(idx[j]),v) for v in range(4) for j in range(ptr[v],ptr[v+1])]
        self.assertEqual(sorted(actual),sorted([(2,1),(2,1),(0,2)]+[(i,i) for i in range(4)]))
        self.assertNotIn((1,2),actual);np.testing.assert_array_equal(eids,np.arange(7,dtype=np.int64))

    def test_1024_row_addresses_group_padding_and_large_byte_offsets(self):
        from .geometry import address,grouped_rows
        self.assertEqual([address(i,8)['byte_in_page'] for i in range(8)],[0,1024,2048,3072]*2)
        self.assertEqual(address(4,8)['nvme_read_offset_bytes'],4096)
        group=grouped_rows(4,8);self.assertEqual([x['row_offset_bytes'] for x in group],[4096,5120])
        value=address(133633039,133633040,3*2**40)
        self.assertEqual(value['row_offset_bytes'],3*2**40+133633039*1024)
        for row in (-1,8):
            with self.assertRaises(ValueError):address(row,8)
        with self.assertRaises(ValueError):grouped_rows(2,8)

    def test_pool_extent_refuses_pa_width_and_unverified_overrun(self):
        from .geometry import extent
        self.assertEqual(extent(8,4096,8192),8192)
        for args in ((7,4096,8192),(8,1,8192),(8,4096,4096),(8,2**63-4096,8192)):
            with self.assertRaises(ValueError):extent(*args)

    def test_256d_loader_parameters_and_shared_proxy_rejection(self):
        from .runtime import loader_kwargs,setup_sampling_imports
        setup_sampling_imports()
        self.assertTrue(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')))
        from digit.io_geometry import IOGeometry
        g=IOGeometry.create(feature_row_bytes=1024,group_size=2,minimum_transfer_bytes=4096,target_request_bytes=4096)
        p=compile_plan();pool=dict(passed=True,feature_mode='logical_synthetic',row_bytes=1024,offset=3*2**40,verified_bytes=8192)
        for arm in ARMS:
            kw=loader_kwargs(p,arm,8,pool,g);self.assertEqual(kw['cache_dim'],256);self.assertEqual(kw['num_ele'],2048)
            self.assertEqual(kw['cache_size'],4096);self.assertEqual(kw['off'],3*2**40)
        for key,value in (('row_bytes',512),('feature_mode','physical_row_proxy'),('passed',False)):
            with self.assertRaises(ValueError):loader_kwargs(p,'gids',8,dict(pool,**{key:value}),g)

    def test_exact_cpu_mapping_aliases_replicas_and_never_caches_padding(self):
        from candidates.pa_sage_cache_policy_v2.tests import FakeStore
        from .backend import ExactInstaller
        from .mapping import install
        p=np.array([0,1,3,5],np.int64);s=np.array([0,1,-1,2,0,3,2,-1],np.int64);hot=np.array([0,2],np.int64)
        for chosen in (hot,np.empty(0,np.int64)):
            fs=FakeStore();arm=dict(cpu_rows=len(chosen),gpu_policy='fifo',gpu_feature_cache_bytes=4*2**30)
            r=install(ExactInstaller(fs),arm,chosen,p,s,chunk_rows=3)
            self.assertEqual(fs.slots,[1,0,0,2,1,0,2,0] if len(chosen) else [0]*8)
            self.assertEqual(r['cpu_feature_bytes'],len(chosen)*1024)
        with self.assertRaises(ValueError):install(ExactInstaller(FakeStore()),arm,chosen,p,s,feature_mode='physical_row_proxy')

    def test_native_installer_rejects_half_width_and_incomplete_maps(self):
        from candidates.pa_sage_cache_policy_v2.tests import FakeStore
        from .backend import ExactInstaller
        b=ExactInstaller(FakeStore())
        with self.assertRaises(ValueError):b.begin_exact_cpu_cache(np.array([0],np.int64),4,512)
        b.begin_exact_cpu_cache(np.array([0],np.int64),4,1024)
        with self.assertRaises(ValueError):b.finish_exact_cpu_cache()
        with self.assertRaises(ValueError):b.write_cpu_row_map(0,np.array([2],np.uint32))
        b.write_cpu_row_map(0,np.array([1,0,0,0],np.uint32));b.finish_exact_cpu_cache();b.configure_gpu_cache('fifo',4*2**30)
        with self.assertRaises(ValueError):b.configure_gpu_cache('fifo',4*2**30)

    def test_1024_byte_native_interval_preserves_512_byte_io_granules(self):
        from candidates.pa_sage_cpu_capacity_v1.tests import counter_fixture
        from .counters import interval
        from .accounting import validate_region,summarize
        for cpu_rows in (0,2):
            before,after=counter_fixture(cpu_rows,60)
            for x in (before,after):
                for key in ('ssd_useful_bytes','gpu_feature_bytes'):x['useful_io'][key]*=2
            r=interval(before,after,60,True);validate_region(r)
            self.assertEqual(r['feature_row_bytes'],1024);self.assertEqual(r['useful_io']['granularity_bytes'],512)
            r['feature_seconds']=1.;summary=summarize([(r,2.)]);self.assertEqual(summary['logical_feature_bytes'],60*1024)
            self.assertEqual(r['device']['completed_bytes'],r['device']['primary_bytes']+r['device']['replay_bytes'])
            bad=copy.deepcopy(after);bad['useful_io']['gpu_feature_bytes']//=2
            with self.assertRaises(RuntimeError):interval(before,bad,60,True)

    def test_counter_rejects_stale_512_byte_region_and_zero_capacity_cpu_hits(self):
        from candidates.pa_sage_cpu_capacity_v1.tests import counter_fixture
        from .counters import interval
        before,after=counter_fixture(0,60)
        with self.assertRaises(RuntimeError):interval(before,after,60,True)
        for x in (before,after):
            for key in ('ssd_useful_bytes','gpu_feature_bytes'):x['useful_io'][key]*=2
        after['feature']['cpu']=1
        with self.assertRaises(ValueError):interval(before,after,61,True)


def run_checks(output):
    state=cpu_gate();os.nice(19);os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(60,65));resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protected=read(OUT/'protected_before.json')
    require(all(sha(ROOT/n)==h for n,h in protected.items()),'Protected files changed before checks')
    started=time.time();capture=io.StringIO()
    for file in HERE.glob('*.py'):ast.parse(file.read_text(),filename=str(file))
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (output/'tests.log').write_text(capture.getvalue());print(capture.getvalue(),flush=True)
    require(result.wasSuccessful(),'CPU unit checks failed')
    cpu_gate()
    from .fixture import exercise
    fixture=exercise(output/'sage_fixture')
    after={n:sha(ROOT/n) for n in protected};require(after==protected,'Active or deferred source/binary changed')
    import torch
    require(not torch.cuda.is_initialized() and not any(k.startswith('BAM_Feature_Store') for k in sys.modules),'CPU fixture imported a feature backend or initialized CUDA')
    final=read(GRID/'status.json')
    summary=dict(passed=True,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        true_cpu_sage_fixture=fixture,wall_seconds=time.time()-started,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.nice(0),protected_files_unchanged=len(after),
        native_execution=False,compilation=False,gpu_queries=False,raw_ssd_access=False,large_preparation=False,
        grid_before={k:state.get(k) for k in ('stage','pid','completed')},grid_after={k:final.get(k) for k in ('stage','pid','completed')})
    write_new(output/'summary.json',summary);write_new(output/'protected_after.json',after)
    print('CPU UKS/SAGE fixture passed; native acceptance remains deferred.',flush=True)
