"""CPU-only tests of the actual transformed loop, source parity and fail-closed gates."""
import ast
import copy
import ctypes
import hashlib
import inspect
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from .common import HERE, ROOT, OUT, cfg, require, setup, sha, write
from .profile import HostProfile, TOP, dump, transform, StripSpans, instrument_function, install, validate

setup()
import dgl
from .model import make_model, optimizer
from . import worker
from .features import NativeFeatures


def tree_function(path, name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)


class RemoveHooks(ast.NodeTransformer):
    def visit_Expr(self,node):
        c=node.value
        if isinstance(c,ast.Call):
            if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='PROFILE':
                return None
            if isinstance(c.func,ast.Name) and c.func.id=='install':
                return None
            if isinstance(c.func,ast.Name) and c.func.id=='write' and "stage_profile.json" in dump(c):
                return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if dump(node.targets[0])==dump(ast.parse("report['stage_profile']",mode='eval').body).replace('Load()', 'Store()', 1):
            return None
        if isinstance(node.targets[0],ast.Subscript) and "stage_profile" in dump(node.targets[0]):
            return None
        return self.generic_visit(node)


class FakeStore:
    def __init__(self, features, values):self.features,self.values=features,values
    def begin_useful_io_region(self):pass
    def get_useful_io_stats(self):return [0,2]
    def read_feature(self,out,ids,count,dim,cache_dim,offset,flags):
        index=np.ctypeslib.as_array((ctypes.c_int64*count).from_address(ids))
        dest=np.ctypeslib.as_array((ctypes.c_float*(count*dim)).from_address(out)).reshape(count,dim)
        dest[:]=self.values[index]


class FakeFeatures(NativeFeatures):
    """Real fetch method and real tensors, but host memory instead of SSD/CUDA."""
    def __init__(self, values):
        self.arm='gids';self.plan=dict(rows=len(values));self.feature_seconds=0.
        self.store=FakeStore(self,values)
    def begin(self):return self.feature_seconds
    def finish(self,before,count):
        return dict(rows=count,feature_seconds=self.feature_seconds-before,feature_cpu=count,feature_gpu_ssd=0)


def run_cpu_loop(mode, folder):
    # Execute the actual nested fetch/phase functions from worker.execute.
    profile=HostProfile(mode)
    execute=tree_function(HERE/'worker.py','execute')
    module=ast.Module(body=[execute],type_ignores=[])
    if mode=='host':module,_=transform(module,'worker')
    functions=[n for n in module.body[0].body if isinstance(n,ast.FunctionDef) and n.name in ('fetch','phase')]
    torch.manual_seed(72);dgl.seed(72)
    graph=dgl.graph((torch.arange(24).repeat_interleave(5), (torch.arange(24).repeat_interleave(5)+torch.arange(5).repeat(24))%24),num_nodes=24)
    sampler=dgl.dataloading.NeighborSampler([10,5,5],replace=False)
    model=make_model('sage');opt=optimizer(model)
    values=np.random.RandomState(42).normal(size=(24,1024)).astype('float32')
    features=FakeFeatures(values);install(profile,sampler,graph,features)
    labels=np.arange(24,dtype='float32')%19
    config=dict(cfg(),batch_size=2)
    def aggregate(windows):
        return dict(useful_io=dict(region_id=2),feature=dict(cpu=sum(w['rows'] for w in windows),gpu_ssd=0))
    ns=dict(np=np,torch=torch,dgl=dgl,hashlib=hashlib,time=time,PROFILE=profile,
        retained=lambda:None,model=model,opt=opt,features=features,sampler=sampler,
        graph=graph,arrays=None,labels=labels,p=config,require=require,
        a=SimpleNamespace(output=folder,arm='gids',model='sage',smoke=False),
        aggregate=aggregate,append_sync=lambda *a,**k:None,progress=lambda *a,**k:None,
        write=write,plan=dict(required_bytes=1024),raw=None,audits=[])
    original_empty=torch.empty
    def empty(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_empty(*args,**kwargs)
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(HERE/'worker.py')+':cpu-fixture','exec'),ns)
        # This fixture replaces existing GPU APIs, never invokes a CUDA runtime.
        with mock.patch.object(torch.Tensor,'cuda',lambda tensor,*a,**k:tensor), \
             mock.patch.object(torch,'empty',side_effect=empty), \
             mock.patch.object(torch.cuda,'synchronize') as synchronize, \
             mock.patch.object(torch.cuda,'mem_get_info',return_value=(10**12,10**12)):
            result=ns['phase']('training',np.arange(8,dtype='int64'),sampler,2)
        return dict(phase=result,profile=profile.result(),model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(opt.state_dict()),synchronize_calls=synchronize.call_count)
    finally:profile.close()


class Checks(unittest.TestCase):
    def test_original_executable_ast_unchanged_except_profile_hooks(self):
        parent=ROOT/'candidates/ig_perf_window300_v1'
        current=RemoveHooks().visit(tree_function(HERE/'worker.py','execute'))
        original=tree_function(parent/'worker.py','execute')
        # Only local candidate import namespaces differ.
        normalized=dump(current).replace('candidates.ig_sage_stage_profile_v1','candidates.ig_perf_window300_v1')
        self.assertEqual(normalized,dump(original))
        self.assertEqual((HERE/'features.py').read_text().replace('candidates.ig_sage_stage_profile_v1','candidates.ig_perf_window300_v1'),(parent/'features.py').read_text())
        self.assertEqual(sha(HERE/'protocol.json'),sha(parent/'protocol.json'))
        self.assertEqual(sha(HERE/'runtime/IGPerfNative.so'),sha(parent/'runtime/IGPerfNative.so'))

    def test_instrumented_ast_strips_to_original_and_keeps_all_sites(self):
        for filename,name,kind in [('worker.py','execute','worker'),('features.py','fetch','features')]:
            original=ast.Module(body=[tree_function(HERE/filename,name)],type_ignores=[])
            instrumented,sites=transform(copy.deepcopy(original),kind)
            self.assertEqual(dump(StripSpans().visit(instrumented)),dump(original))
            self.assertTrue(all(n==1 for n in sites.values()))
        broken=ast.Module(body=[tree_function(HERE/'worker.py','execute')],type_ignores=[])
        for n in ast.walk(broken):
            if isinstance(n,ast.Attribute) and n.attr=='backward':n.attr='renamed_backward'
        with self.assertRaisesRegex(RuntimeError,'instrumentation sites'):transform(broken,'worker')

    def test_actual_loop_loss_gradients_adam_and_sync_count_match_on_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=run_cpu_loop('off',root/'off');b=run_cpu_loop('host',root/'host')
        self.assertEqual(a['phase']['losses'],b['phase']['losses'])
        self.assertEqual(a['phase']['sampling_shape_totals'],b['phase']['sampling_shape_totals'])
        self.assertEqual(a['synchronize_calls'],b['synchronize_calls'])
        for key,value in a['model'].items():self.assertTrue(torch.equal(value,b['model'][key]))
        for key,state in a['optimizer']['state'].items():
            self.assertEqual(int(state['step']),4)
            for k,value in state.items():self.assertTrue(torch.equal(value,b['optimizer']['state'][key][k]))
        for result,mode in [(a,'off'),(b,'host')]:
            r=dict(stage_profile=result['profile'],training=result['phase'],warmup=None,arm='gids')
            self.assertTrue(validate(r))
            self.assertEqual(r['stage_profile']['mode'],mode)

    def test_nested_time_excludes_children_and_no_off_clock(self):
        p=HostProfile('host',clock=iter([0,10,30,50]).__next__);p.start_window()
        with p.span('parent'):
            with p.span('child'):pass
        p.finish_window('training',dict(index=0,batches=1,seconds=100e-9))
        w=p.windows[0];self.assertEqual(w['spans']['parent']['exclusive_seconds'],30e-9)
        self.assertEqual(w['top_level_seconds'],50e-9)
        p=HostProfile('off',clock=mock.Mock(side_effect=AssertionError('off clock called')));p.start_window()
        with p.span('disabled'):pass
        p.finish_window('training',dict(index=0,batches=1,seconds=1.));self.assertEqual(p.windows[0]['spans'],{})

    def test_profile_rejects_missing_adam_and_timing_double_count(self):
        with tempfile.TemporaryDirectory() as td:result=run_cpu_loop('host',Path(td))
        r=dict(stage_profile=result['profile'],training=result['phase'],warmup=None,arm='gids')
        for fault in ('missing_adam','double_count','sync'):
            wrong=copy.deepcopy(r)
            if fault=='missing_adam':del wrong['stage_profile']['windows'][0]['spans']['adam']
            elif fault=='double_count':wrong['stage_profile']['windows'][0]['top_level_seconds']=10000.
            else:wrong['stage_profile']['added_cuda_synchronization']=True
            with self.assertRaises(RuntimeError):validate(wrong)

    def test_symmetric_plan_and_smoke_failures_block_all_formal_runs(self):
        from . import controller as c
        jobs=c.plan('representative')
        plan=json.loads((HERE/'profile_protocol.json').read_text())['schedule']
        self.assertEqual([(j['mode'],j['arm'],j['profile_mode']) for j in jobs],[tuple(j[:3]) for j in plan])
        for arm in ('gids','digit_full'):
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
                seen=[]
                def run(o,j,*args):
                    seen.append(j)
                    if j['arm']==arm:raise RuntimeError('fixture smoke failed')
                    return {}
                def bind(o,e):write(o/'inputs.json',{});return {}
                with mock.patch.object(c,'verify',return_value='candidate'),mock.patch('candidates.ig_sage_stage_profile_v1.common.verify',return_value='entry'),mock.patch.object(c,'input_binding',side_effect=bind),mock.patch.object(c,'run_worker',side_effect=run):
                    with self.assertRaisesRegex(RuntimeError,'fixture smoke failed'):c.execute('representative',Path(td)/'fresh',2,'sage')
                self.assertTrue(all(j['mode']=='smoke' for j in seen))

    def test_monitor_exit_terminates_worker_before_acceptance(self):
        from . import controller as c
        child=mock.Mock(pid=998);child.poll.return_value=None
        monitor=mock.Mock();monitor.process.poll.return_value=2;monitor.start.return_value=dict(pid=997,passed=True);monitor.stop.return_value=2
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            out=Path(td);state=dict(pid=996,workers=[]);job=c.plan('smoke')[0]
            with mock.patch.object(c,'check_device'),mock.patch.object(c,'ExternalMonitor',return_value=monitor),mock.patch.object(c.subprocess,'Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'monitor exited during worker'):c.run_worker(out,job,state,{},lambda **kw:None,2)
            child.terminate.assert_called_once();monitor.stop.assert_called_once()
            self.assertFalse((out/'smoke_gids_accepted.json').exists())

    def test_digit_wrapper_descriptors_restore_and_call_once(self):
        from uva_sampler import UVAMetadata, PinnedBuffer
        p=HostProfile('host');p.start_window()
        with mock.patch.object(UVAMetadata,'sample',autospec=True,return_value='sample') as sample, \
             mock.patch.object(PinnedBuffer,'__getitem__',autospec=True,return_value='gather') as gather:
            original_sample=UVAMetadata.sample;original_gather=PinnedBuffer.__getitem__
            p.wrap(UVAMetadata,'sample','uva_sample_existing_waits')
            p.wrap(PinnedBuffer,'__getitem__','uva_address_gather_existing_waits')
            meta=UVAMetadata.__new__(UVAMetadata);buffer=object.__new__(PinnedBuffer)
            with p.span('sampling'):
                with p.span('outer_frontier'):self.assertEqual(meta.sample('roots',10,0),'sample')
                with p.span('storage_annotation'):self.assertEqual(buffer['roots'],'gather')
            sample.assert_called_once_with(meta,'roots',10,0);gather.assert_called_once_with(buffer,'roots')
            p.close();self.assertIs(UVAMetadata.sample,original_sample);self.assertIs(PinnedBuffer.__getitem__,original_gather)
        p.finish_window('training',dict(index=0,batches=1,seconds=10.))
        self.assertIn('sampling/outer_frontier/uva_sample_existing_waits',p.windows[0]['spans'])
        self.assertIn('sampling/storage_annotation/uva_address_gather_existing_waits',p.windows[0]['spans'])

    def test_summary_footer_keeps_negative_overhead_nested_times_and_numerical_differences(self):
        from .summarize import emit
        value=dict(instrumentation_overhead={a:dict(off_seconds=2.,host_seconds=1.5,difference_seconds=-.5) for a in ('gids','digit_full')},
            stages=[dict(stage='forward',nested=False,gids_seconds=.4,digit_seconds=.5,digit_minus_gids_seconds=.1),
                    dict(stage='sampling/to_block',nested=True,gids_seconds=.1,digit_seconds=.2,digit_minus_gids_seconds=.1)],
            windows=[dict(arm=a,profile_mode='host',phase='training',uninstrumented_seconds=.01) for a in ('gids','digit_full')],
            same_arm_comparisons={a:dict(all_losses_exact=False,final_parameters_exact=False,adam_state_exact=False,
                max_abs_loss_difference=.001,max_abs_parameter_difference=.0001,sampled_shape_totals_equal=True) for a in ('gids','digit_full')},
            limitations=['fixture limitation'])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'summary';emit(value,out);text=(out/'README.md').read_text()
            self.assertIn('Host minus off | -0.50 | -0.50',text)
            self.assertIn('exact final parameters=False',text)
            self.assertIn('already included above',text)
            self.assertEqual(json.loads((out/'summary.json').read_text()),value)
            self.assertEqual(len((out/'stages.csv').read_text().splitlines()),3)

    def test_export_failure_marks_run_incomplete_without_repeating_workers(self):
        from . import controller as c
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            out=Path(td)/'fresh';seen=[]
            def bind(o,e):write(o/'inputs.json',{});return {}
            def run(o,j,*args):
                seen.append(j);write(o/(j['mode']+'_'+j['arm']+'_accepted.json'),{})
                return dict(external_monitor=dict(strict_monitor_passed=True))
            with mock.patch.object(c,'verify',return_value='candidate'), \
                 mock.patch('candidates.ig_sage_stage_profile_v1.common.verify',return_value='candidate'), \
                 mock.patch.object(c,'input_binding',side_effect=bind),mock.patch.object(c,'check_inputs'), \
                 mock.patch.object(c,'check_device'),mock.patch.object(c,'run_worker',side_effect=run), \
                 mock.patch.object(c,'pair_check',return_value=dict(passed=True)), \
                 mock.patch('candidates.ig_sage_stage_profile_v1.summarize.load_formal',side_effect=RuntimeError('fixture export failure')):
                with self.assertRaisesRegex(RuntimeError,'fixture export failure'):c.execute('representative',out,2)
            self.assertEqual(len(seen),6)
            state=json.loads((out/'status.json').read_text());self.assertFalse(state['passed']);self.assertFalse(state['complete'])
            self.assertEqual(state['stage'],'summary_failed')


if __name__=='__main__':
    torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    raise SystemExit(not result.wasSuccessful())
