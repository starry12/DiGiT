"""Bounded CPU tests. All GPU jobs, raw I/O, compilation and real monitors are excluded."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from .common import ROOT,HERE,OUT,GRID,read,sha,write,identity,require,ARMS
from .protocol import compile_plan,schedule
from .controller import Journal,stage_key
from .protocol import arms
from .profile import collect_batches


class FakeMonitor:
    """Synthetic files only; never calls nvidia-smi or claims native validation."""
    fail=False
    def __init__(self,output):
        self.output=Path(output);self.pid=os.getpid()+100000;self.stopped=False
        self.process=self
    def poll(self):return 0 if self.stopped else None
    def start(self):
        self.output.mkdir(parents=True)
        return dict(passed=True,pid=self.pid)
    def stop(self):
        if self.stopped:return 1 if self.fail else 0
        self.stopped=True
        sample=dict(time_unix=time.time(),monitor_pid=self.pid,monitor_parent_pid=os.getpid(),device_used_bytes=0)
        (self.output/'samples.jsonl').write_text(json.dumps(sample)+'\n')
        write(self.output/'summary.json',dict(fixture=True,passed=not self.fail,complete=True,pid=self.pid,
            parent_pid=os.getpid(),peak_rss_bytes=1024,errors=['injected monitor error'] if self.fail else []))
        return 1 if self.fail else 0


FAKE_WORKER = '''import sys,json,os,time,hashlib
from pathlib import Path
o=Path(sys.argv[1]);o.mkdir()
r={'passed':True,'native':False,'fixture':True,'kind':'cpu_monitor_fixture'}
(o/'report.json').write_text(json.dumps(r))
resources={'mode':'checkpoints_only_external_sampler','background_monitor_in_worker':False,
 'monitor_samples':0,'worker_pid':os.getpid(),'observed_peak_device_used_bytes':0}
(o/'resources.json').write_text(json.dumps(resources))
(o/'worker_ready.json').write_text(json.dumps({'passed':True,'report_sha256':hashlib.sha256((o/'report.json').read_bytes()).hexdigest()}))
deadline=time.monotonic()+8
while not (o/'release_worker.json').exists():
 if time.monotonic()>deadline:raise RuntimeError('fixture handshake timed out')
 time.sleep(.02)
assert json.loads((o/'release_worker.json').read_text())['passed']
'''


class Tests(unittest.TestCase):
    def test_active_grid_gate_precedes_mutations_processes_and_device_imports(self):
        from .controller import run
        with patch('candidates.pa_sage_cpu_capacity_v1.common.read',return_value=dict(complete=False,passed=False)), \
             patch('subprocess.Popen') as popen,patch('pathlib.Path.mkdir') as mkdir:
            with self.assertRaises(ValueError):run(Path('/unused/protocol'),Path('/unused/output'))
            popen.assert_not_called();mkdir.assert_not_called()

    def test_service_launcher_also_refuses_active_grid_before_systemd(self):
        from .start import main
        with patch('sys.argv',['start','--execute']), \
             patch('candidates.pa_sage_cpu_capacity_v1.common.read',return_value=dict(complete=False,passed=False)), \
             patch('subprocess.run') as child,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):main()
            child.assert_not_called()

    def test_schedule_places_all_short_acceptances_before_any_full_worker(self):
        items=schedule(compile_plan());keys=[stage_key(x) for x in items]
        self.assertEqual(keys[:4],['bind','build','profile','prepare'])
        self.assertEqual(keys[4:8],['smoke_'+a for a in ARMS])
        self.assertEqual(keys[8:12],['full_'+a for a in ARMS]);self.assertEqual(keys[-1],'aggregate')

    def test_protocol_rejects_changed_model_cache_capacity_or_monitor_timing(self):
        from .protocol import validate
        from .protocol import arms
        p=compile_plan()
        for change in ('cache','model','monitor','profile'):
            bad=copy.deepcopy(p)
            if change=='cache':bad['arms']=arms(20)
            elif change=='model':bad['hidden']=64
            elif change=='monitor':bad['execution']['monitor_query_timeout_seconds']=30
            else:bad['profile']['batches']=99
            with self.assertRaises(ValueError):validate(bad)

    def test_frequency_counts_batch_inputs_and_rejects_leakage_in_extent(self):
        counts=np.zeros(6,np.int64)
        r=collect_batches([np.array([0,1,4],np.int64),np.array([1,4],np.int64)],counts,2)
        np.testing.assert_array_equal(counts,[1,2,0,0,2,0]);self.assertEqual(r['total_logical_requests'],5)
        self.assertEqual(r['optimizer_updates'],0)
        for batches,k in [([np.array([0,0],np.int64)],1),([np.array([-1],np.int64)],1),
                          ([np.array([6],np.int64)],1),([np.array([1],np.int64)],2),
                          ([np.array([1],np.int64),np.array([2],np.int64)],1)]:
            with self.assertRaises(ValueError):collect_batches(batches,np.zeros(6,np.int64),k)

    def test_journal_preserves_failed_attempt_and_detects_tampered_accepted_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';j=Journal(root,'a','b');attempt=j.attempt('rank')
            write(attempt/'report.json',dict(passed=True,fixture=True));j.commit('rank',attempt/'report.json')
            failed=j.attempt('profile');write(failed/'failure.json',dict(fixture=True));j.fail(RuntimeError('injected'))
            resumed=Journal(root,'a','b',True);self.assertEqual(resumed.previous('rank'),attempt/'report.json')
            new=resumed.attempt('profile');self.assertNotEqual(new,failed);self.assertTrue((failed/'failure.json').exists())
            write(attempt/'report.json',dict(passed=True,fixture=True,changed=True))
            with self.assertRaises(ValueError):resumed.previous('rank')
            with self.assertRaises(ValueError):Journal(root,'different','b',True)

    def test_monitor_handshake_uses_separate_owned_process_and_retains_fixture_label(self):
        from .monitor import run_monitored
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp)/'attempt';attempt.mkdir();events=[]
            report,row=run_monitored([sys.executable,'-B','-c',FAKE_WORKER,str(attempt/'worker')],attempt,'fixture',
                lambda **kw:events.append(kw),monitor_factory=FakeMonitor)
            self.assertEqual(row['returncode'],0);self.assertTrue(report['fixture']);self.assertFalse(report['native'])
            self.assertNotEqual(row['pid'],os.getpid());self.assertTrue(read(attempt/'worker/release_worker.json')['passed'])

    def test_monitor_failure_never_releases_success_and_reaps_only_own_worker(self):
        from .monitor import run_monitored
        class Failing(FakeMonitor):fail=True
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp)/'attempt';attempt.mkdir();events=[]
            with self.assertRaises(ValueError):run_monitored(
                [sys.executable,'-B','-c',FAKE_WORKER,str(attempt/'worker')],attempt,'fixture',
                lambda **kw:events.append(kw),monitor_factory=Failing)
            self.assertFalse((attempt/'worker/release_worker.json').exists())
            pid=events[0]['pid']
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_worker_exit_before_ready_cannot_be_accepted(self):
        from .monitor import run_monitored
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp)/'attempt';attempt.mkdir()
            with self.assertRaises(ValueError):run_monitored(
                [sys.executable,'-B','-c','raise SystemExit(2)'],attempt,'fixture',monitor_factory=FakeMonitor)
            self.assertFalse((attempt/'monitored.json').exists())

    def test_controller_failure_resume_reuses_completed_stages_and_never_starts_full_early(self):
        from . import controller
        p=compile_plan();events=[];failure={'enabled':True}
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as patches:
            root=Path(tmp);protocol=root/'protocol.json';write(protocol,p);binary=root/'dummy.so';binary.write_bytes(b'fixture')
            manifest=read(Path(p['layout']['base'])/'final/bundle/manifest.json')
            def fake_bind(p,protocol,out,execution):
                write(out/'inputs.json',dict(passed=True,fixture=True,manifest=manifest));events.append('bind')
            def fake_plain(command,attempt,notify):
                events.append('rank');(attempt/'worker').mkdir();write(attempt/'worker/report.json',dict(passed=True,fixture=True))
            def fake_monitored(command,attempt,arm,notify):
                mode=command[command.index('--mode')+1];events.append(mode+'_'+arm)
                if mode=='smoke' and arm=='cpu10' and failure['enabled']:raise RuntimeError('injected smoke interruption')
                if mode=='full':self.assertTrue((root/'run/short_matrix.json').exists())
                return dict(passed=True,fixture=True,kind='independent_native_presampling' if mode=='profile' else 'fixture',
                    optimizer_updates=0,evaluation_calls=0,feature_reads=0),dict(returncode=0)
            def fake_seal(p,protocol,binding,profile,path,execution):
                events.append('prepare');write(path,dict(passed=True,fixture=True))
            def fake_matrix(paths,*args):
                self.assertEqual(set(paths),set(ARMS))
                return dict(passed=True,fixture=True,reports={a:dict(path=str(path),sha256=sha(path)) for a,path in paths.items()})
            def fake_aggregate(paths,*args):
                self.assertEqual(set(paths),set(ARMS));events.append('aggregate')
                return dict(passed=True,fixture=True,arms={a:dict(training_seconds=1,order_excluded_seconds=1,
                    speedup_vs_cpu00=1,cpu_percent=0,cpu_rows=0,cpu_feature_bytes=1,gpu_feature_cache_bytes=0) for a in ARMS})
            targets={
                'controller.heavy_gate':lambda:None,'controller.verify':lambda:'source',
                'controller.lock':lambda *a,**k:contextlib.nullcontext(),
                'controller.LARGE':root/'large','controller.run_plain':fake_plain,'controller.run_monitored':fake_monitored,
                'controller.wait_available':lambda *a:None,'binding.bind':fake_bind,'binding.check':lambda *a:None,
                'prepared.seal':fake_seal,'prepared.check':lambda *a:None,
                'validation.short':lambda *a:None,'validation.full':lambda *a:None,
                'validation.make_smoke_matrix':fake_matrix,'validation.smoke_matrix':lambda *a:None,
                'validation.aggregate':fake_aggregate}
            for name,value in targets.items():patches.enter_context(patch('candidates.pa_sage_cpu_capacity_v1.'+name,value))
            patches.enter_context(patch('candidates.pa_sage_cpu_capacity_v1.common.BINARY',binary))
            patches.enter_context(patch('candidates.pa_sage_cpu_capacity_v1.common.binary_receipt',return_value={'binary_sha256':'fixture'}))
            patches.enter_context(patch('ae.common.host',return_value=256*2**30))
            with self.assertRaises(RuntimeError):controller.run(protocol,root/'run')
            self.assertFalse(any(e.startswith('full_') for e in events))
            self.assertFalse(read(root/'run/status.json')['complete'])
            failure['enabled']=False;controller.run(protocol,root/'run',resume=True)
            state=read(root/'run/status.json');self.assertTrue(state['complete'] and state['passed'])
            self.assertEqual(len(state['completed']),13)
            self.assertEqual(events.count('profile_profile'),1)
            self.assertEqual(events.count('smoke_cpu10'),2)
            self.assertTrue((root/'run/stages/smoke_cpu10/attempt_00').exists())
            self.assertTrue((root/'run/stages/smoke_cpu10/attempt_01').exists())


    def test_capacity_denominator_exact_rows_and_gpu_fixed(self):
        p=compile_plan();n=p['graph']['nodes']
        self.assertEqual([p['arms'][a]['cpu_rows'] for a in ARMS],[0,5552997,11105995,22211991])
        for percent,arm in zip((0,5,10,20),ARMS):
            row=p['arms'][arm];self.assertEqual(row['cpu_feature_bytes'],(n*percent//100)*512)
            self.assertEqual(row['gpu_feature_cache_bytes'],4*2**30);self.assertEqual(row['gpu_policy'],'fifo')
        self.assertEqual(arms(19)['cpu05']['cpu_rows'],0)
        self.assertFalse(p['scope']['paper_matrix_complete'])

    def test_nested_topk_matches_stable_full_sort_including_empty_and_ties(self):
        from .selection import select_all,nested
        p=compile_plan();p['graph']['nodes']=100;p['arms']=arms(100)
        counts=np.random.default_rng(9).integers(0,7,size=100,dtype=np.int64)
        expected=sorted(range(100),key=lambda i:(-counts[i],i))
        with tempfile.TemporaryDirectory() as tmp:
            descriptors=select_all(counts,p,Path(tmp))
            hot={a:np.load(d['path']) for a,d in descriptors.items()}
            for a in ARMS:np.testing.assert_array_equal(hot[a],sorted(expected[:p['arms'][a]['cpu_rows']]))
            self.assertTrue(nested(hot,p))
            hot['cpu05'][0]=-1
            with self.assertRaises(ValueError):nested(hot,p)

    def test_zero_and_nonzero_installer_replicas_padding_and_invalid_zero_slot(self):
        from .backend import ExactInstaller
        from candidates.pa_sage_cache_policy_v2.tests import FakeStore
        from candidates.pa_sage_cache_policy_v1.native_adapter import install
        storage=np.array([0,1,-1,2,0,3,2,-1],np.int64);primary=np.array([0,1,3,5],np.int64)
        for nodes in ([],[0,2]):
            hot=np.array(nodes,np.int64);arm=dict(arms(20)['cpu00'],cpu_rows=len(hot),cpu_feature_bytes=len(hot)*512)
            fs=FakeStore();installer=ExactInstaller(fs)
            receipt=install(installer,arm,hot,primary,storage,chunk_rows=3)
            self.assertEqual(fs.calls[0],('begin',[] if not nodes else [0,3],8))
            self.assertEqual(fs.slots,[0]*8 if not nodes else [1,0,0,2,1,0,2,0])
            self.assertEqual(fs.calls[-1],('configure',2,4*2**30))
            self.assertEqual(receipt['cpu_feature_bytes'],512*len(nodes))
            self.assertEqual(receipt['row_map_gpu_bytes'],32)
        fs=FakeStore();installer=ExactInstaller(fs)
        installer.begin_exact_cpu_cache(np.empty(0,np.int64),8,512)
        with self.assertRaises(ValueError):installer.write_cpu_row_map(0,np.array([1],np.uint32))
        with self.assertRaises(ValueError):installer.finish_exact_cpu_cache()
        self.assertEqual(fs.calls,[('begin',[],8)])

    def test_all_four_loader_paths_identical_and_proxy_aliasing_rejected(self):
        from .runtime import loader_kwargs
        p=compile_plan();m=dict(grouping=dict(group_size=2),feature=dict(row_bytes=512,dim=128,num_storage_rows=24),io=dict(page_size=4096))
        class Geometry:
            def with_payload_offset(self,offset):self.offset=offset;return self
            def to_native_mapping(self):return dict(offset=self.offset)
        kwargs=[loader_kwargs(p,a,m,Geometry()) for a in ARMS]
        self.assertTrue(all(k==kwargs[0] for k in kwargs));self.assertEqual(kwargs[0]['cache_size'],4096)
        p['features']['mode']='physical_row_proxy'
        with self.assertRaises(ValueError):loader_kwargs(p,'cpu00',m,Geometry())

    def test_single_profile_seal_nested_and_rejects_training_leakage(self):
        from .selection import select_all
        from .prepared import seal,check
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=compile_plan();p['graph']['nodes']=100;p['arms']=arms(100)
            protocol=root/'protocol.json';write(protocol,p);binding=root/'binding.json'
            write(binding,dict(graph_sha256='g',layout_sha256='l'))
            counts=root/'counts.npy';values=np.arange(100,dtype=np.int64);np.save(counts,values)
            hot=select_all(values,p,root)
            profile=root/'profile.json'
            r=dict(passed=True,fixture=False,source_sha256='source',protocol_sha256=sha(protocol),binding_sha256=sha(binding),
                graph_sha256='g',layout_sha256='l',kind='independent_native_presampling',native=True,worker_returncode=0,
                monitor=dict(passed=True,backend='nvidia-smi'),seed=23,batches=100,optimizer_updates=0,evaluation_calls=0,
                feature_reads=0,raw_ssd_access=False,hot=hot,fanouts=p['fanouts'],batch_size=p['batch_size'],
                counts=dict(path=str(counts),sha256=sha(counts),identity=identity(counts)))
            write(profile,r);out=root/'prepared.json';value=seal(p,protocol,binding,profile,out,'source')
            self.assertTrue(value['nested_hot_sets']);self.assertEqual(value['arms']['cpu00']['rows'],0)
            check(value,p,protocol,binding,'source')
            for key,val in [('seed',0),('batches',99),('optimizer_updates',1),('feature_reads',1)]:
                write(profile,dict(r,**{key:val}))
                with self.assertRaises(ValueError):seal(p,protocol,binding,profile,out,'source')

    def test_zero_cpu_counters_preserve_real_io_partition_and_reject_false_cpu_hit(self):
        from .counters import interval
        for rows in (0,5):
            before,after=counter_fixture(rows,60)
            region=interval(before,after,60,True)
            self.assertEqual(region['cpu_rows'],rows)
            self.assertEqual(region['serving']['cpu_served_rows'],0 if rows==0 else 20)
            self.assertGreater(region['serving']['gpu_hit_rows'],0);self.assertGreater(region['device']['primary_bytes'],0)
            self.assertEqual(region['device']['completed_bytes'],region['device']['primary_bytes']+region['device']['replay_bytes'])
        before,after=counter_fixture(0,60);after['feature']['cpu']=1
        with self.assertRaises(ValueError):interval(before,after,61,True)

    def test_cpu_oracle_feature_and_sgd_equivalence_across_capacities(self):
        # Tiny serial route/value oracle, NOT a native SAGE benchmark.
        from candidates.pa_sage_cache_policy_v1.oracle import FeatureOracle
        from candidates.pa_sage_cache_policy_v1.selection import topk
        n=100;rng=np.random.default_rng(1);features=rng.normal(size=(n,128)).astype(np.float32)
        storage=np.r_[np.arange(n),np.arange(n),np.full(8,-1)].astype(np.int64)
        payload=np.zeros((len(storage),128),np.float32);valid=storage>=0;payload[valid]=features[storage[valid]]
        batches=[np.unique(np.r_[np.arange(5),rng.integers(0,n,size=10)]).astype(np.int64) for _ in range(12)]
        reference=None;reports={}
        for a,settings in arms(n).items():
            hot=topk(np.arange(n,0,-1,dtype=np.int64),settings['cpu_rows'])
            oracle=FeatureOracle(storage,payload,features,hot,4096) # scaled one-page FIFO for the fixture
            weights=np.full(128,.001,np.float32)
            for step,ids in enumerate(batches):
                rows=ids+(n if step%2 else 0);x=oracle.fetch(ids,rows)
                np.testing.assert_array_equal(x,features[ids])
                target=ids.astype(np.float32)/n;residual=x.dot(weights)-target
                weights-=.0001*x.T.dot(residual)/len(ids)
            if reference is None:reference=weights.copy()
            np.testing.assert_array_equal(weights,reference)
            reports[a]=oracle.report();self.assertEqual(reports[a]['logical_requests'],sum(map(len,batches)))
            self.assertGreater(reports[a]['ssd_served_rows'],0)
        self.assertEqual(reports['cpu00']['cpu_served_rows'],0)
        self.assertGreater(reports['cpu20']['cpu_served_rows'],0)
        self.assertEqual(len(set(r['storage_request_sha256'] for r in reports.values())),1)

    def test_smoke_matrix_matches_features_and_requires_all_four_before_full(self):
        from .validation import make_smoke_matrix,smoke_matrix
        p=compile_plan()
        with tempfile.TemporaryDirectory() as tmp:
            paths={}
            for a in ARMS:
                r=report_fixture(p,a);r.update(kind='native_cpu_capacity_short',updates=4,
                    feature_bit_exact=True,sample_edges_verified=True,exact_smoke_trace_sha256='1'*64,exact_smoke_features_sha256='2'*64)
                path=Path(tmp)/(a+'.json');write(path,r);paths[a]=path
            matrix=make_smoke_matrix(paths,p,'a'*64,'b'*64,'prep','binary')
            smoke_matrix(matrix,p,'a'*64,'b'*64,'prep')
            bad=read(paths['cpu10']);bad['exact_smoke_features_sha256']='3'*64;write(paths['cpu10'],bad)
            with self.assertRaises(ValueError):make_smoke_matrix(paths,p,'a'*64,'b'*64,'prep','binary')
            with self.assertRaises(ValueError):make_smoke_matrix({'cpu00':paths['cpu00']},p,'a'*64,'b'*64,'prep','binary')

    def test_full_capacity_summary_and_reject_short_wrong_budget_or_changed_workload(self):
        from .validation import full,aggregate
        p=compile_plan()
        with tempfile.TemporaryDirectory() as tmp:
            short=Path(tmp)/'short.json';write(short,dict(fixture=True));paths={}
            for i,a in enumerate(ARMS):
                r=report_fixture(p,a);r['native_short_receipt_sha256']=sha(short)
                r['training_seconds']=2+i # Deliberately slower with more CPU; never enforce the paper's outcome.
                path=Path(tmp)/(a+'.json');write(path,r);paths[a]=path
                full(r,p,a,'a'*64,'b'*64,'prep',short,'binary')
            result=aggregate(paths,p,'a'*64,'b'*64,'prep',short,'binary')
            self.assertEqual(result['arms']['cpu20']['speedup_vs_cpu00'],2/5)
            self.assertEqual(result['arms']['cpu00']['cpu_feature_bytes'],0)
            for key,val in [('updates',4),('examples',1),('fixture',True),('feature_seconds',0),('cpu_rows',1)]:
                bad=read(paths['cpu00']);bad[key]=val
                with self.assertRaises(ValueError):full(bad,p,'cpu00','a'*64,'b'*64,'prep',short,'binary')
            bad=read(paths['cpu10']);bad['sample_trace_sha256']='9'*64;write(paths['cpu10'],bad)
            with self.assertRaises(ValueError):aggregate(paths,p,'a'*64,'b'*64,'prep',short,'binary')

    def test_build_and_import_gate_block_while_grid_active(self):
        from . import build,runtime
        for target,call in [('build.read',build.execute),('runtime.read',runtime.prepare_imports)]:
            with patch('candidates.pa_sage_cpu_capacity_v1.'+target,return_value=dict(complete=False,passed=False)), \
                 patch('subprocess.run') as run,patch('importlib.import_module') as imp,patch('pathlib.Path.mkdir') as mkdir:
                with self.assertRaises(ValueError):call()
                run.assert_not_called();imp.assert_not_called();mkdir.assert_not_called()


def counter_fixture(cpu_rows,logical_rows):
    from candidates.pa_sage_cache_policy_v2.tests import counters
    b=counters(2);a=copy.deepcopy(b)
    for x in (b,a):
        x['policy'].update(cpu_rows=cpu_rows,preload_rows=cpu_rows,storage_rows=412470240,gpu_feature_bytes=4*2**30)
        x['gpu']['capacity_pages']=2**20
    cpu=logical_rows//3 if cpu_rows else 0;route=logical_rows-cpu;hits=route//3;fills=route-hits
    a['feature']=dict(cpu=cpu,gpu_ssd=route)
    a['gpu'].update(requests=route,hits=hits,inserts=fills,physical_inserts=fills,fifo_ticket=fills,
        resident_pages=min(fills,2**20),evictions=max(0,fills-2**20))
    a['device'].update(submitted_commands=fills*2,completed_commands=fills*2,completed_bytes=fills*4608,
        active_ns=fills*100,total_latency_ns=fills*200,max_latency_ns=100,max_outstanding=1,replay_commands=fills,replay_bytes=fills*512)
    a['useful_io'].update(ssd_fill_bytes=fills*4096,ssd_useful_bytes=fills*512,gpu_feature_bytes=route*512,gpu_feature_rows=route)
    return b,a


def report_fixture(p,arm):
    from .counters import interval
    from .acceptance import TRACE_KEYS
    examples=p['execution']['training_examples'];rows=p['arms'][arm]['cpu_rows']
    region=interval(*counter_fixture(rows,examples*2),examples*2,True)
    r=dict(kind='native_cpu_capacity_full_epoch',native=True,fixture=False,passed=True,source_only=False,
        arm=arm,worker_returncode=0,source_sha256='a'*64,protocol_sha256='b'*64,prepared_sha256='prep',backend_binary_sha256='binary',
        native_short_receipt_sha256='c'*64,epochs=1,updates=1179,evaluation='disabled',finite_loss_and_gradients=True,
        training_seconds=1.,order_excluded_seconds=.9,feature_seconds=.5,cpu_cache_install_seconds=0 if not rows else .1,setup_seconds=.2,
        cpu_rows=rows,hot_nodes_sha256='d'*64,monitor=dict(passed=True,backend='nvidia-smi'),region=region,examples=examples,
        windows=[dict(start_update=1,end_update=1179,updates=1179,region=region)],
        shapes=[dict(output_nodes=min(p['batch_size'],examples-i*p['batch_size']),input_nodes=2*min(p['batch_size'],examples-i*p['batch_size'])) for i in range(1179)])
    r.update({k:'e'*64 for k in TRACE_KEYS});return r


def run_checks(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('OMP_NUM_THREADS')=='1' and
            os.environ.get('OPENBLAS_NUM_THREADS')=='1','CPU checks need hidden CUDA and one thread')
    state=read(GRID/'status.json')
    require(not state['stage'].startswith(('native_','budget_')),'Defer CPU checks during native grid measurement')
    os.nice(19);os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(60,65));resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    before=read(OUT/'protected_before.json')
    require(all(sha(ROOT/name)==digest for name,digest in before.items()),'Protected source changed before checks')
    start=time.time()
    for file in HERE.glob('*.py'):ast.parse(file.read_text(),filename=str(file))
    capture=io.StringIO();result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (output/'tests.log').write_text(capture.getvalue());print(capture.getvalue(),flush=True)
    after={name:sha(ROOT/name) for name in before}
    require(before==after,'Active grid or frozen backend files changed')
    require(not any(name=='torch' or name.startswith('BAM_Feature_Store') for name in sys.modules),'CPU rehearsal imported device runtime')
    summary=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        fixture=True,evidence_kind='CPU arrays, synthetic counters, mocked controller jobs and stdlib child processes',
        native_execution=False,heavy_compilation=False,large_graph_preparation=False,gpu_queries=False,raw_ssd_access=False,
        monitor_backend_exercised='synthetic fixture; real nvidia-smi deferred',
        protected_files_unchanged=len(after),wall_seconds=time.time()-start,
        peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.nice(0),
        grid_before={k:state.get(k) for k in ('stage','pid','completed')},
        grid_after={k:read(GRID/'status.json').get(k) for k in ('stage','pid','completed')})
    write(output/'summary.json',summary);write(output/'protected_after.json',after)
    require(result.wasSuccessful(),'CPU controller checks failed')
