"""CPU-only regressions for paired scope, affinity, lifecycle and result recovery."""
import ast,copy,shutil,tempfile,unittest
from pathlib import Path
from unittest import mock
import torch
from .common import HERE,ROOT,setup,read,write,sha,POLICY
setup()
SOURCE=ROOT/'results/ig_sage_stage_profile_20260926_v1/native/20260926-225054-3269104/experiment'


class Checks(unittest.TestCase):
    def test_original_training_sampling_model_and_io(self):
        parent=ROOT/'candidates/ig_sage_stage_profile_v1'
        def fn(path,name):return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('fetch','phase'):
            self.assertEqual(ast.dump(fn(HERE/'worker.py',name)),ast.dump(fn(parent/'worker.py',name)))
        for name in ('model.py','features.py','profile.py','windows.py','protocol.json','admission.py','validation.py'):
            self.assertEqual((HERE/name).read_text().replace(HERE.name,parent.name),(parent/name).read_text())
        self.assertEqual(sha(HERE/'runtime/IGPerfNative.so'),sha(parent/'runtime/IGPerfNative.so'))
        source=(HERE/'worker.py').read_text()
        self.assertIn('from uva_sampler import UVANeighborSampler',source)
        self.assertIn('from sampler_config import configure',source)
        from sampler_config import expected_configuration
        for arm in ('gids','full'):
            self.assertIn('IGGroupWarpCUDA',expected_configuration(arm)['module']['path'])
        from .profile import HostProfile,install
        # An off profile must not inspect or patch sampling/features.
        install(HostProfile('off'),None,None,None)

    def test_only_digit_threads_are_bound(self):
        from . import affinity as a
        before=dict(thread_masks={'101':[0,1,2],'102':[0,1,2]})
        after=dict(thread_masks={'101':[2],'102':[2]})
        with mock.patch.object(a,'snapshot',side_effect=[before,after]),mock.patch.object(a.os,'sched_getaffinity',return_value={0,1,2}),mock.patch.object(a.os,'sched_setaffinity') as setter:
            self.assertEqual(a.apply_affinity('digit_full')['policy'],'one_core')
            self.assertEqual(setter.call_args_list,[mock.call(101,{2}),mock.call(102,{2})])
        with mock.patch.object(a,'snapshot',return_value=before),mock.patch.object(a.os,'sched_setaffinity') as setter:
            initial=a.apply_affinity('gids');setter.assert_not_called()
            self.assertEqual(initial['policy'],'unchanged')
        with self.assertRaisesRegex(RuntimeError,'GIDS default CPU mask changed'):
            a.validate_affinity(dict(initial=initial,final=after),'gids')
        with mock.patch.object(a,'snapshot',return_value=before),mock.patch.object(a.os,'sched_getaffinity',return_value={0,1}):
            with self.assertRaises(RuntimeError):a.apply_affinity('digit_full')

    def test_two_smokes_then_off_abba(self):
        from .controller import plan,worker_command
        jobs=plan();self.assertEqual([j['arm'] for j in jobs],['gids','digit_full','gids','digit_full','digit_full','gids'])
        self.assertEqual([j['smoke'] for j in jobs],[True,True,False,False,False,False])
        self.assertTrue(all(j['profile_mode']=='off' for j in jobs))
        self.assertEqual(len({j['mode'] for j in jobs}),6)
        self.assertTrue(all('--variant' not in worker_command(j,Path('/tmp/fixture')) for j in jobs))
        self.assertEqual(read(HERE/'protocol.json')['measured_batches'],300)

    def test_either_failed_smoke_prevents_formal(self):
        from . import controller as c
        for fails_at in (1,2):
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
                def bind(o,e):write(o/'inputs.json',{});return {}
                effects=[{'fixture':True}]*(fails_at-1)+[RuntimeError('fixture smoke failed')]
                with mock.patch.object(c,'verify',return_value='candidate'),mock.patch.object(c,'input_binding',side_effect=bind),mock.patch.object(c,'run_worker',side_effect=effects) as worker:
                    with self.assertRaisesRegex(RuntimeError,'fixture smoke failed'):c.execute('representative',Path(td)/'fresh',2)
                    self.assertEqual(worker.call_count,fails_at)
                    self.assertFalse(read(Path(td)/'fresh/status.json')['passed'])

    def test_monitor_death_terminates_worker(self):
        from . import controller as c
        child=mock.Mock(pid=998);child.poll.return_value=None
        mon=mock.Mock();mon.process.poll.return_value=2;mon.start.return_value=dict(passed=True);mon.stop.return_value=2
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            with mock.patch.object(c,'check_device'),mock.patch.object(c,'ExternalMonitor',return_value=mon),mock.patch.object(c.subprocess,'Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'monitor exited'):c.run_worker(Path(td),c.plan()[0],dict(pid=997,workers=[]),{},lambda **kw:None,2)
            child.terminate.assert_called_once();mon.stop.assert_called_once()

    def test_actual_version_and_cross_system_pairing(self):
        from .summarize import verify_version
        from .validation import pair_check
        for key in ('launch','state','binding'):
            v={k:dict(candidate_sha256='current') for k in ('launch','state','binding')}
            verify_version(v['launch'],v['state'],v['binding'],'current');v[key]['candidate_sha256']='old'
            with self.assertRaises(RuntimeError):verify_version(v['launch'],v['state'],v['binding'],'current')
        reports={a:read(SOURCE/'control'/a/'report.json') for a in ('gids','digit_full')}
        self.assertNotEqual(reports['gids']['training']['losses'],reports['digit_full']['training']['losses'])
        self.assertTrue(pair_check(reports,False)['passed'])
        bad=copy.deepcopy(reports);bad['digit_full']['training']['roots_sha256']='wrong'
        with self.assertRaisesRegex(RuntimeError,'Unpaired roots'):pair_check(bad,False)
        bad=copy.deepcopy(reports);bad['digit_full']['model_config']['dropout']=-1
        with self.assertRaisesRegex(RuntimeError,'Unpaired model_config'):pair_check(bad,False)

    def test_complete_receipts_summary_roundtrip_and_tamper(self):
        from . import controller as c,review as rev,summarize as s
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            out=Path(td);jobs=c.plan();workers=[];token='fixture-current';reports={}
            write(out/'inputs.json',dict(candidate_sha256=token,files={}))
            for j in jobs:
                source=SOURCE/('smoke' if j['smoke'] else 'control')/j['arm']
                original=read(source/'report.json')
                original['external_monitor']=read(SOURCE/(('smoke' if j['smoke'] else 'control')+'_'+j['arm']+'_accepted.json'))['external_monitor']
                original['affinity']=dict(initial=dict(policy='one_core' if j['arm']=='digit_full' else 'unchanged'))
                folder=out/j['mode']/j['arm'];folder.mkdir(parents=True)
                write(folder/'report.json',original);reports[j['mode']]=original
                shutil.copyfile(source/'final_model.pt',folder/'final_model.pt')
                prefix=j['mode']+'_'+j['arm'];(out/(prefix+'.log')).write_text('fixture\n');write(out/(prefix+'_accepted.json'),original)
                write(out/(prefix+'_receipt.json'),dict(passed=True,job=j,files=c.arm_evidence(out,j),accepted_sha256=sha(out/(prefix+'_accepted.json')),report_sha256=sha(folder/'report.json')))
                workers.append(dict(j,status='complete',returncode=0,command=c.worker_command(j,out)))
            write(out/'comparisons.json',s.comparisons(reports,out))
            write(out/'status.json',dict(schema='digit-ig-affinity-pair-run-v1',passed=True,complete=True,mode='representative',dataset='IG',model='sage',gpu=2,seed=0,pid=99,candidate_sha256=token,workers=workers,raw_ssd_writes=False,strict_resource_acceptance=True,training_performance_io_passed=True))
            write(out/'launch.json',dict(schema='digit-ig-affinity-pair-launch-v1',candidate_sha256=token,mode='representative',dataset='IG',model='sage',gpu=2,seed=0,input_binding_sha256=sha(out/'inputs.json'),protocol_sha256=sha(HERE/'protocol.json'),optimization_protocol_sha256=sha(HERE/'optimization_protocol.json'),plan=jobs,monitor_policy=POLICY,raw_ssd_writes=False))
            def review(base,w,*args):return read(base/w['mode']/w['arm']/'report.json')
            with mock.patch.object(s,'verify',return_value=token),mock.patch.object(rev,'review_worker',side_effect=review):
                value=s.load_formal(out);s.emit(value,out/'profile_summary');self.assertEqual(s.load_formal(out),read(out/'profile_summary/summary.json'))
                self.assertEqual(len(value['rows']),4);self.assertEqual(value['independent_runs_per_arm'],2)
                self.assertEqual(value['means']['gids']['training_seconds'],reports['gids_1']['training_seconds'])
                self.assertTrue(value['comparisons']['same_arm_repeats']['digit_full']['adam_exact'])
                negative=copy.deepcopy(value);negative['ratios']['mean_training_speedup']=.5
                s.emit(negative,out/'negative');self.assertIn('0.500x',(out/'negative/README.md').read_text())
                (out/'digit_2_digit_full.log').write_text('tampered\n')
                with self.assertRaisesRegex(RuntimeError,'Changed/incomplete evidence'):s.load_formal(out)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main(verbosity=2)
