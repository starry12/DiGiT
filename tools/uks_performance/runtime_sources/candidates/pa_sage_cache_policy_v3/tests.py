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
from .rank import symmetric_revpr,owner_chunks
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
        with patch('candidates.pa_sage_cache_policy_v3.common.read',return_value=dict(complete=False,passed=False)), \
             patch('subprocess.Popen') as popen,patch('pathlib.Path.mkdir') as mkdir:
            with self.assertRaises(ValueError):run(Path('/unused/protocol'),Path('/unused/output'))
            popen.assert_not_called();mkdir.assert_not_called()

    def test_service_launcher_also_refuses_active_grid_before_systemd(self):
        from .start import main
        with patch('sys.argv',['start','--execute']), \
             patch('candidates.pa_sage_cache_policy_v3.common.read',return_value=dict(complete=False,passed=False)), \
             patch('subprocess.run') as child,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):main()
            child.assert_not_called()

    def test_schedule_places_all_short_acceptances_before_any_full_worker(self):
        items=schedule(compile_plan());keys=[stage_key(x) for x in items]
        self.assertEqual(keys[:5],['bind','build','rank','profile','prepare'])
        self.assertEqual(keys[5:9],['smoke_'+a for a in ARMS])
        self.assertEqual(keys[9:13],['full_'+a for a in ARMS]);self.assertEqual(keys[-1],'aggregate')

    def test_protocol_rejects_changed_model_cache_capacity_or_monitor_timing(self):
        from .protocol import validate
        from candidates.pa_sage_cache_policy_v1.protocol import arms
        p=compile_plan()
        for change in ('cache','model','monitor','profile'):
            bad=copy.deepcopy(p)
            if change=='cache':bad['arms']=arms(2,4*2**30)
            elif change=='model':bad['hidden']=64
            elif change=='monitor':bad['execution']['monitor_query_timeout_seconds']=30
            else:bad['profile']['batches']=99
            with self.assertRaises(ValueError):validate(bad)

    def test_segmented_revpr_matches_dense_reverse_walk_with_multiedges_and_dangling(self):
        from candidates.pa_sage_cache_policy_v1.selection import reverse_pagerank
        pairs=[(0,1),(0,1),(1,0),(1,0),(1,2),(2,1),(2,2),(3,4),(4,3)]
        n=6;adj=np.zeros((n,n))
        for u,v in pairs:adj[u,v]+=1
        idx=np.array([u for v in range(n) for u,w in pairs if w==v],np.int64)
        ptr=np.r_[0,np.cumsum([sum(w==v for u,w in pairs) for v in range(n)])].astype(np.int64)
        expected=np.full(n,1/n);trans=adj.T.copy()
        for i,row in enumerate(trans):trans[i]=row/row.sum() if row.sum() else 1/n
        for _ in range(20):expected=.15/n+.85*trans.T.dot(expected)
        with tempfile.TemporaryDirectory() as tmp:
            degree,scores=symmetric_revpr(ptr,idx,Path(tmp)/'ranks')
            np.testing.assert_array_equal(degree,np.diff(ptr))
            np.testing.assert_allclose(scores,expected,rtol=0,atol=1e-14)
            np.testing.assert_allclose(scores,reverse_pagerank(ptr,idx),rtol=0,atol=1e-14)

    def test_owner_chunks_cover_empty_and_high_degree_nodes_once(self):
        ptr=np.array([0,0,0,20,20,21,21,21],np.int64)
        ranges=list(owner_chunks(ptr,max_nodes=2,max_edges=3))
        self.assertEqual([i for lo,hi in ranges for i in range(lo,hi)],list(range(len(ptr)-1)))
        self.assertTrue(all(hi>lo for lo,hi in ranges))

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

    def test_profile_seal_shares_one_freq_file_and_rejects_measured_profile(self):
        from .prepared import seal,check
        from candidates.pa_sage_cache_policy_v1.protocol import arms
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=root/'protocol.json';binding=root/'binding.json'
            p=compile_plan();p['arms']=arms(2,4*2**30);write(protocol,p)
            write(binding,dict(graph_sha256='g',layout_sha256='l'))
            hot={}
            for name in ('degree','revpr','freq'):
                path=root/(name+'.npy');np.save(path,np.array([0,2],np.int64))
                hot[name]=dict(path=str(path),sha256=sha(path),identity=identity(path),rows=2)
            counts=root/'counts.npy';np.save(counts,np.array([5,0,5],np.int64))
            shared=dict(passed=True,fixture=False,source_sha256='source',protocol_sha256=sha(protocol),
                        binding_sha256=sha(binding),graph_sha256='g')
            ranks=root/'ranks.json';write(ranks,dict(shared,hot={k:hot[k] for k in ('degree','revpr')}))
            profile=root/'profile.json';pr=dict(shared,kind='independent_native_presampling',native=True,
                worker_returncode=0,monitor=dict(passed=True,backend='nvidia-smi'),seed=23,batches=100,
                optimizer_updates=0,evaluation_calls=0,feature_reads=0,raw_ssd_access=False,hot=hot['freq'],
                layout_sha256='l',fanouts=p['fanouts'],batch_size=p['batch_size'],
                counts=dict(path=str(counts),sha256=sha(counts),identity=identity(counts)))
            write(profile,pr)
            out=root/'prepared.json';value=seal(p,protocol,binding,ranks,profile,out,'source')
            self.assertEqual(value['arms']['freq'],value['arms']['digit'])
            check(value,p,protocol,binding,'source')
            for key,val in [('seed',0),('optimizer_updates',1),('batches',99),('feature_reads',1)]:
                write(profile,dict(pr,**{key:val}))
                with self.assertRaises(ValueError):seal(p,protocol,binding,ranks,profile,out,'source')

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
                if mode=='smoke' and arm=='freq' and failure['enabled']:raise RuntimeError('injected smoke interruption')
                if mode=='full':self.assertTrue((root/'run/short_matrix.json').exists())
                return dict(passed=True,fixture=True,kind='independent_native_presampling' if mode=='profile' else 'fixture',
                    optimizer_updates=0,evaluation_calls=0,feature_reads=0),dict(returncode=0)
            def fake_seal(p,protocol,binding,ranks,profile,path,execution):
                events.append('prepare');write(path,dict(passed=True,fixture=True))
            def fake_matrix(paths,*args):
                self.assertEqual(set(paths),set(ARMS))
                return dict(passed=True,fixture=True,reports={a:dict(path=str(path),sha256=sha(path)) for a,path in paths.items()})
            def fake_aggregate(paths,*args):
                self.assertEqual(set(paths),set(ARMS));events.append('aggregate')
                return dict(passed=True,fixture=True,arms={a:dict(training_seconds=1,order_excluded_seconds=1,
                    speedup_vs_freq=1,cpu_feature_bytes=1,gpu_feature_cache_bytes=0) for a in ARMS})
            targets={
                'controller.heavy_gate':lambda:None,'controller.verify':lambda:'source',
                'controller.lock':lambda *a,**k:contextlib.nullcontext(),
                'controller.LARGE':root/'large','controller.run_plain':fake_plain,'controller.run_monitored':fake_monitored,
                'controller.wait_available':lambda *a:None,'binding.bind':fake_bind,'binding.check':lambda *a:None,
                'prepared.seal':fake_seal,'prepared.check':lambda *a:None,
                'validation.short':lambda *a:None,'validation.full':lambda *a:None,
                'validation.make_smoke_matrix':fake_matrix,'validation.smoke_matrix':lambda *a:None,
                'validation.aggregate':fake_aggregate}
            for name,value in targets.items():patches.enter_context(patch('candidates.pa_sage_cache_policy_v3.'+name,value))
            patches.enter_context(patch('candidates.pa_sage_cache_policy_v2.common.BINARY',binary))
            patches.enter_context(patch('candidates.pa_sage_cache_policy_v2.common.binary_receipt',return_value={'binary_sha256':'fixture'}))
            patches.enter_context(patch('ae.common.host',return_value=256*2**30))
            with self.assertRaises(RuntimeError):controller.run(protocol,root/'run')
            self.assertFalse(any(e.startswith('full_') for e in events))
            self.assertFalse(read(root/'run/status.json')['complete'])
            failure['enabled']=False;controller.run(protocol,root/'run',resume=True)
            state=read(root/'run/status.json');self.assertTrue(state['complete'] and state['passed'])
            self.assertEqual(len(state['completed']),14)
            self.assertEqual(events.count('rank'),1);self.assertEqual(events.count('profile_profile'),1)
            self.assertEqual(events.count('smoke_freq'),2)
            self.assertTrue((root/'run/stages/smoke_freq/attempt_00').exists())
            self.assertTrue((root/'run/stages/smoke_freq/attempt_01').exists())

    def test_full_acceptance_rejects_short_or_incomplete_training_coverage(self):
        from .validation import full
        from candidates.pa_sage_cache_policy_v2.tests import report,counters
        from candidates.pa_sage_cache_policy_v2.counters import interval
        p=compile_plan();examples=p['execution']['training_examples'];factor=examples//3
        before,after=counters(),counters(end=True)
        for k in after['feature']:after['feature'][k]*=factor
        after['policy']['bypass_ssd_rows']*=factor
        for k in ('submitted_commands','completed_commands','completed_bytes','active_ns','total_latency_ns','replay_commands','replay_bytes'):
            after['device'][k]*=factor
        for k in ('ssd_fill_bytes','ssd_useful_bytes','gpu_feature_bytes','gpu_feature_rows'):after['useful_io'][k]*=factor
        region=interval(before,after,6*factor,True)
        with tempfile.TemporaryDirectory() as tmp:
            short=Path(tmp)/'short.json';write(short,dict(fixture=True))
            r=report('degree');r.update(fixture=False,prepared_sha256='prep',backend_binary_sha256='binary',
                native_short_receipt_sha256=sha(short),examples=examples,cpu_rows=p['arms']['degree']['cpu_rows'],
                feature_seconds=1,region=region,windows=[dict(start_update=1,end_update=1179,updates=1179,region=region)],
                shapes=[dict(output_nodes=min(p['batch_size'],examples-i*p['batch_size']),
                             input_nodes=2*min(p['batch_size'],examples-i*p['batch_size'])) for i in range(1179)])
            full(r,p,'degree','a'*64,'b'*64,'prep',short,'binary')
            for key,val in [('updates',4),('examples',examples-1),('fixture',True),('feature_seconds',0)]:
                bad=copy.deepcopy(r);bad[key]=val
                with self.assertRaises(ValueError):full(bad,p,'degree','a'*64,'b'*64,'prep',short,'binary')


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
