"""CPU checks for stable frontier compaction, affinity scope and report closure."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from .common import HERE,ROOT,setup,sha,read,write
setup()
import dgl
from .sampler import compact_columns,SharedIndexSampler
from uva_sampler import UVANeighborSampler


class Checks(unittest.TestCase):
    def test_compaction_covers_empty_padding_duplicates_and_large_eids(self):
        for n in (0,1,9):
            for fanout in (2,10):
                for pattern in ('none','all','mixed'):
                    length=n*fanout;src=torch.arange(length,dtype=torch.int64)%7
                    if pattern=='none':src[:]=-1
                    if pattern=='mixed':src[::3]=-1
                    rows=torch.arange(length,dtype=torch.int64)+2**32;flags=(rows%2).byte();eids=rows+2**34;seeds=torch.arange(n,dtype=torch.int64)*3
                    mask=src>=0;want=(src[mask],seeds.repeat_interleave(fanout)[mask],rows[mask],flags[mask],eids[mask])
                    got=compact_columns(src,rows,flags,eids,seeds,fanout)
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(want,got)))

    def test_full_frontier_structure_and_seed_counter_match(self):
        seeds=torch.tensor([1,4,7]);fanout=3;sources=torch.tensor([3,-1,5,1,1,5,0,2,-1]);rows=torch.arange(9)+2**32
        flags=torch.tensor([0,0,1,1,0,0,0,1,0],dtype=torch.uint8);eids=torch.arange(9)+2**34
        groups=torch.tensor([1,1,1]);nodes=torch.tensor([0,1,0]);results=[]
        for method in (UVANeighborSampler._cuda_group_frontier,SharedIndexSampler._cuda_group_frontier):
            metadata=SimpleNamespace(sample=mock.Mock(return_value=(sources,rows,flags,eids,groups,nodes)))
            obj=SimpleNamespace(num_nodes=10,random_seed=11,_cuda_call_counter=3,_ensure_cuda_metadata=lambda *args:metadata)
            f,g,n=method(obj,None,seeds,fanout);metadata.sample.assert_called_once_with(seeds,fanout,14)
            self.assertEqual(obj._cuda_call_counter,4);self.assertIs(g,groups);self.assertIs(n,nodes)
            results.append(f)
        for a,b in zip(results[0].edges(),results[1].edges()):self.assertTrue(torch.equal(a,b))
        for key in results[0].edata:self.assertTrue(torch.equal(results[0].edata[key],results[1].edata[key]))
        for fmt in ('csc','csr'):
            for a,b in zip(results[0].adj_tensors(fmt),results[1].adj_tensors(fmt)):self.assertTrue(torch.equal(a,b))

    def test_actual_training_and_feature_code_unchanged(self):
        parent=ROOT/'candidates/ig_sage_stage_profile_v1'
        def func(path,name):return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('fetch','phase'):
            self.assertEqual(ast.dump(func(HERE/'worker.py',name),include_attributes=False),ast.dump(func(parent/'worker.py',name),include_attributes=False))
        for name in ('model.py','features.py','profile.py','windows.py','protocol.json'):
            self.assertEqual((HERE/name).read_text().replace('candidates.ig_sage_host_opt_v1','candidates.ig_sage_stage_profile_v1'),(parent/name).read_text())
        self.assertEqual(sha(HERE/'runtime/IGPerfNative.so'),sha(parent/'runtime/IGPerfNative.so'))
        from .profile import transform
        for name,fn,kind in [('worker.py','execute','worker'),('features.py','fetch','features')]:
            _,sites=transform(ast.Module(body=[func(HERE/name,fn)],type_ignores=[]),kind)
            self.assertTrue(all(n==1 for n in sites.values()))

    def test_affinity_only_touches_current_workers_threads(self):
        from . import affinity as a
        unbound=dict(thread_masks={'101':[0,1,2,3],'102':[0,1,2,3]},cpu2_frequency_khz=1)
        bound=dict(thread_masks={'101':[2],'102':[2]},cpu2_frequency_khz=1)
        for variant in ('legacy','compact'):
            with mock.patch.object(a,'snapshot',return_value=unbound),mock.patch.object(a.os,'sched_setaffinity') as set_mask:
                self.assertEqual(a.apply_affinity(variant)['policy'],'unchanged');set_mask.assert_not_called()
        with mock.patch.object(a,'snapshot',side_effect=[unbound,bound]),mock.patch.object(a.os,'sched_getaffinity',return_value={0,1,2,3}),mock.patch.object(a.os,'sched_setaffinity') as set_mask:
            self.assertEqual(a.apply_affinity('combined')['policy'],'one_core')
            self.assertEqual(set_mask.call_args_list,[mock.call(101,{2}),mock.call(102,{2})])
        with mock.patch.object(a,'snapshot',return_value=unbound),mock.patch.object(a.os,'sched_getaffinity',return_value={0,1}):
            with self.assertRaises(RuntimeError):a.apply_affinity('affinity')

    def test_actual_worker_manifest_checked_instead_of_parent(self):
        from .summarize import verify_version
        launch=dict(candidate_sha256='current');state=dict(candidate_sha256='current');binding=dict(candidate_sha256='current')
        verify_version(launch,state,binding,'current')
        for obj in (launch,state,binding):
            obj['candidate_sha256']='parent'
            with self.assertRaises(RuntimeError):verify_version(launch,state,binding,'current')
            obj['candidate_sha256']='current'

    def test_smoke_failure_prevents_formal_and_probe(self):
        from . import controller as c
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            def bind(o,e):write(o/'inputs.json',{});return {}
            with mock.patch.object(c,'verify',return_value='candidate'),mock.patch.object(c,'input_binding',side_effect=bind), \
                 mock.patch.object(c,'run_worker',side_effect=RuntimeError('fixture smoke failed')) as worker, \
                 mock.patch.object(c.subprocess,'call') as probe:
                with self.assertRaisesRegex(RuntimeError,'fixture smoke failed'):c.execute('representative',Path(td)/'fresh',2)
                self.assertEqual(worker.call_count,1);self.assertTrue(worker.call_args[0][1]['smoke']);probe.assert_not_called()

    def test_live_monitor_failure_stops_worker(self):
        from . import controller as c
        child=mock.Mock(pid=998);child.poll.return_value=None
        mon=mock.Mock();mon.process.poll.return_value=2;mon.start.return_value=dict(passed=True);mon.stop.return_value=2
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            with mock.patch.object(c,'check_device'),mock.patch.object(c,'ExternalMonitor',return_value=mon),mock.patch.object(c.subprocess,'Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'monitor exited'):c.run_worker(Path(td),c.plan()[0],dict(pid=997,workers=[]),{},lambda **kw:None,2)
            child.terminate.assert_called_once();mon.stop.assert_called_once()

    def test_real_checkpoint_equivalence_and_changed_model_rejected(self):
        from .summarize import compare_reports
        folder=ROOT/'results/ig_sage_stage_profile_20260926_v1/native/20260926-225054-3269104/experiment/profile/digit_full'
        a=read(folder/'report.json');a['variant']='legacy';b=dict(a,variant='compact')
        value=compare_reports(a,b,folder,folder)
        self.assertTrue(value['losses_exact'] and value['final_parameters_exact'] and value['adam_exact'])
        with tempfile.TemporaryDirectory() as td:
            ck=torch.load(folder/'final_model.pt',map_location='cpu')
            first=next(iter(ck['model']));ck['model'][first]=ck['model'][first]+.1;torch.save(ck,Path(td)/'final_model.pt')
            with self.assertRaisesRegex(RuntimeError,'State tensor values differ'):compare_reports(a,b,folder,Path(td))

    def test_export_keeps_all_variants_and_negative_result(self):
        from .summarize import emit
        rows=[dict(variant=v,seconds=10.,sampling_seconds=2.,feature_fetch_seconds=5.,forward_seconds=1.,backward_seconds=1.,adam_seconds=.1,speedup=.9,
            windows=[3.,3.,4.],exact_losses=True,exact_parameters=True,exact_adam=True) for v in ('legacy','affinity','combined','compact')]
        value=dict(rows=rows,sampling_probe=dict(group_selection_ms=1.,eid_resolution_ms=2.,native_total_ms=3.),limits=['fixture'])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'summary';emit(value,out)
            self.assertEqual(read(out/'summary.json'),value)
            self.assertEqual(len((out/'summary.csv').read_text().splitlines()),5)
            text=(out/'README.md').read_text();self.assertIn('0.90',text);self.assertIn('Not training E2E',text)

    def test_complete_reader_and_footer_on_bound_filesystem_fixture(self):
        from . import controller as c,review as rev,summarize as s
        import shutil
        source=ROOT/'results/ig_sage_stage_profile_20260926_v1/native/20260926-225054-3269104/experiment/profile/digit_full'
        original=read(source/'report.json');original['external_monitor']=dict(strict_monitor_passed=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            out=Path(td);jobs=c.plan();workers=[];token='current-fixture-version'
            write(out/'inputs.json',dict(candidate_sha256=token,files={}))
            for j in jobs:
                folder=out/j['mode']/j['arm'];folder.mkdir(parents=True)
                r=dict(original,variant=j['variant']);write(folder/'report.json',r)
                shutil.copyfile(source/'final_model.pt',folder/'final_model.pt')
                prefix=j['mode']+'_'+j['arm'];(out/(prefix+'.log')).write_text('fixture\n')
                write(out/(prefix+'_accepted.json'),r)
                write(out/(prefix+'_receipt.json'),dict(passed=True,job=j,files=c.arm_evidence(out,j),
                    accepted_sha256=sha(out/(prefix+'_accepted.json')),report_sha256=sha(folder/'report.json')))
                workers.append(dict(j,status='complete',returncode=0,command=c.worker_command(j,out)))
                if not j['smoke'] and j['variant']!='legacy':
                    value=s.compare_reports(dict(original,variant='legacy'),r,out/'legacy/digit_full',folder)
                    write(out/(j['mode']+'_equivalence.json'),value)
            state=dict(schema='digit-ig-host-opt-run-v1',passed=True,complete=True,mode='representative',dataset='IG',model='sage',gpu=2,seed=0,pid=99,
                candidate_sha256=token,workers=workers,raw_ssd_writes=False,strict_resource_acceptance=True,training_performance_io_passed=True)
            write(out/'status.json',state)
            write(out/'launch.json',dict(schema='digit-ig-host-opt-launch-v1',candidate_sha256=token,mode='representative',dataset='IG',model='sage',gpu=2,seed=0,
                input_binding_sha256=sha(out/'inputs.json'),protocol_sha256=sha(HERE/'protocol.json'),optimization_protocol_sha256=sha(HERE/'optimization_protocol.json'),
                plan=jobs,monitor_policy=s.POLICY,raw_ssd_writes=False))
            events=[dict(batch=21+i,group_sample_ms=1.,resolve_eids_ms=2.,total_ms=3.) for i in range(8)]
            probe=dict(passed=True,candidate_sha256=token,input_binding_sha256=sha(out/'inputs.json'),training_updates=0,feature_fetches=0,raw_ssd_access=False,
                warmup_samples=20,measured_samples=8,native_events=events,host_records=[{} for _ in events],group_selection_ms=8.,eid_resolution_ms=16.,native_total_ms=24.)
            write(out/'sampling_probe/report.json',probe);(out/'sampling_probe.log').write_text('fixture probe\n')
            write(out/'sampling_probe_receipt.json',dict(passed=True,returncode=0,
                command=['python','-B','-u','-m','candidates.ig_sage_host_opt_v1.sampling_probe','--output',str(out/'sampling_probe'),'--binding',str(out/'inputs.json')],
                report_sha256=sha(out/'sampling_probe/report.json'),log_sha256=sha(out/'sampling_probe.log')))
            def review(base,w,*args):return read(base/w['mode']/w['arm']/'report.json')
            with mock.patch.object(s,'verify',return_value=token),mock.patch.object(rev,'review_worker',side_effect=review):
                value=s.load_formal(out);s.emit(value,out/'profile_summary');self.assertEqual(s.load_formal(out),read(out/'profile_summary/summary.json'))
                self.assertEqual(value['variants'],['legacy','affinity','combined','compact'])
                (out/'compact_digit_full.log').write_text('tampered fixture\n')
                with self.assertRaisesRegex(RuntimeError,'Changed/incomplete evidence'):s.load_formal(out)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main(verbosity=2)
