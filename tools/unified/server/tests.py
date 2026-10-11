import copy
import importlib
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import cli
import install

def result(dataset='UKS', mode='short1'):
    n = 5 if mode == 'short5' else 1
    ratios = [2.11, 1.80, 1.72, 1.69, 1.86][:n]
    return dict(schema='digit-unified-results-v2', dataset=dataset, mode=mode, passed=True, complete=True,
        evidence_scope='native', accuracy_evaluated=dataset=='PA' and mode=='full20',
        paired_rounds=[dict(round=i+1, gids_seconds=x, digit_seconds=1, speedup=x) for i,x in enumerate(ratios)],
        speedup=1.80 if n==5 else ratios[0],
        aggregation='median_of_five_paired_speedups' if n==5 else 'single_pair_ratio_of_total_training_time',
        arms={a:dict(cpu_feature_bytes=100, gpu_feature_bytes=10) for a in ('gids','digit')},
        epochs=[dict(arm=a, round=r, epoch=e, measured_batches=300) for a in ('gids','digit')
                for r in range(1,n+1) for e in range(1,21 if mode=='full20' else 2)])

class Routing(unittest.TestCase):
    def test_fifteen_modes(self):
        for d in common.DATASETS:
            for m in common.MODES:
                self.assertEqual(cli.route(['performance',d,'sage','--mode',m]),d)
                self.assertEqual(cli.route(['results',d,'sage','--action=performance','--mode',m]),d)
    def test_old_pa_routes(self):
        for cmd in ('run','check','smoke','ablation','layout'):
            self.assertIsNone(cli.route([cmd,'PA','sage']))
        self.assertIsNone(cli.route(['results','PA','sage','--action','run']))
    def test_old_model_and_reference_routes(self):
        for d in common.DATASETS:
            for model in ('gcn','gat'):
                self.assertIsNone(cli.route(['performance',d,model]))
        self.assertIsNone(cli.route(['results','IG','sage','--action','performance','--reference']))
        self.assertIsNone(cli.route(['results','UKS','sage','--action','performance','--legacy']))
    def test_cannot_legacy_launch_or_mix_modes(self):
        for args in (['performance','UKS','sage','--legacy'], ['results','IG','sage','--reference','--mode=short1']):
            with self.assertRaises(ValueError): cli.route(args)
    def test_closed_gate_never_starts(self):
        with patch.object(cli,'read',return_value={'datasets':{'UKS':{'enabled':False,'reason':'pending'}}}), patch.object(cli.subprocess,'run') as call:
            self.assertEqual(cli.main(['performance','UKS','sage']),69)
            call.assert_not_called()
    def test_default_median_mode_and_explicit_full20(self):
        with patch.object(cli,'read',return_value={'datasets':{'UKS':{'enabled':True}}}), patch.object(cli.subprocess,'run') as call:
            cli.main(['performance','UKS','sage'])
            self.assertEqual(call.call_args.args[0][-1],common.unit('UKS','short5'))
            cli.main(['performance','UKS','sage','--mode','full20'])
            self.assertEqual(call.call_args.args[0][-1],common.unit('UKS','full20'))

class Results(unittest.TestCase):
    def test_all_modes_and_accuracy_scope(self):
        for d in common.DATASETS:
            for m in common.MODES: common.validate_result(result(d,m),d,m)
    def test_actual_accepted_uks_summary(self):
        p=common.ROOT/'results/unified_experiments_20261011_v2/UKS/runs/20261011-155228-1471501/summary.json'
        if not p.exists(): self.skipTest('Historical author result is not part of installed package')
        self.assertGreater(common.validate_result(common.read(p),'UKS','short1'),0)
    def test_max_is_rejected(self):
        r=result(mode='short5'); r['speedup']=2.11
        with self.assertRaises(RuntimeError): common.validate_result(r,'UKS','short5')
    def test_missing_round_rejected(self):
        r=result(mode='short5'); r['paired_rounds'].pop()
        with self.assertRaises(RuntimeError): common.validate_result(r,'UKS','short5')
    def test_duplicate_epoch_rejected(self):
        r=result(); r['epochs'][-1]=r['epochs'][0]
        with self.assertRaises(RuntimeError): common.validate_result(r,'UKS','short1')
    def test_wrong_scope_or_capacity_rejected(self):
        for field,value in (('accuracy_evaluated',True),('complete',False),('evidence_scope','mock')):
            r=result(); r[field]=value
            with self.assertRaises(RuntimeError): common.validate_result(r,'UKS','short1')
        r=result(); r['arms']['digit']['gpu_feature_bytes']=11
        with self.assertRaises(RuntimeError): common.validate_result(r,'UKS','short1')
    def test_invalid_nonfinite_time_rejected(self):
        r=result(); r['paired_rounds'][0]['gids_seconds']=float('nan')
        with self.assertRaises(RuntimeError): common.validate_result(r,'UKS','short1')

class State(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        self.scope=patch.object(common,'OUT',self.out); self.scope.start(); self.addCleanup(self.scope.stop)
        self.folder=self.out/'UKS/short1/runs/run1'; self.folder.mkdir(parents=True)
        common.write(self.out/'UKS/short1/latest.json',dict(output=str(self.folder),invocation_id='1'*32))
        common.write(self.folder/'summary.json',result())
        common.write(self.folder/'completion.json',dict(invocation_id='1'*32,passed=True,returncode=0,summary_sha256=common.sha(self.folder/'summary.json')))
        self.state=dict(ActiveState='inactive',Result='success',ExecMainStatus='0',InvocationID='1'*32)
    def test_success_requires_receipt_and_exit(self):
        self.assertEqual(common.view('UKS','short1',self.state)['state'],'PASS')
        self.state['ExecMainStatus']='1'
        self.assertEqual(common.view('UKS','short1',self.state)['state'],'FAILED')
    def test_running_never_shows_old_success(self):
        self.state.update(ActiveState='active',InvocationID='2'*32)
        v=common.view('UKS','short1',self.state)
        self.assertEqual(v['state'],'STARTING'); self.assertNotIn('summary',v)
    def test_stale_invocation_and_tamper(self):
        self.state['InvocationID']='2'*32
        self.assertEqual(common.view('UKS','short1',self.state)['state'],'STALE')
        self.state['InvocationID']='1'*32
        common.write(self.folder/'summary.json',{'passed':True})
        with self.assertRaises(RuntimeError): common.view('UKS','short1',self.state)
    def test_error_receipt_wins(self):
        common.write(self.folder/'error.json',{'error':'failed'})
        self.assertEqual(common.view('UKS','short1',self.state)['state'],'FAILED')
    def test_result_path_escape(self):
        common.write(self.out/'UKS/short1/latest.json',dict(output='/etc',invocation_id='1'*32))
        with self.assertRaises(RuntimeError): common.view('UKS','short1',self.state)

class Deployment(unittest.TestCase):
    def test_fixed_units_and_sudoers(self):
        lines=install.sudoers_text().decode().splitlines()
        self.assertEqual(len(lines),30); self.assertEqual(len(set(lines)),30)
        self.assertNotIn('*',install.sudoers_text().decode())
        for d in common.DATASETS:
            for m in common.MODES:
                s=install.service_text(d,m).decode()
                self.assertIn('ConditionPathExists='+str(common.ADMIN/'enabled'/d),s)
                self.assertIn('Restart=no',s); self.assertIn('KillMode=mixed',s)
                self.assertIn(str(common.ADMIN/'runner.py')+' '+d+' '+m,s)
    def test_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'source.py'; p.write_text('x=1\n')
            entry=root/'manifest.json'; entry.write_text('{}')
            binding=dict(entry_sha256=common.sha(entry),files={str(p):common.sha(p)})
            with patch.object(common,'ROOT',root),patch.object(common,'ENTRY',entry):
                common.verify_binding(binding); p.write_text('x=2\n')
                with self.assertRaises(RuntimeError): common.verify_binding(binding)
    def test_rollback_restores_only_owned_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); admin=root/'admin'; admin.mkdir()
            old=root/'entry'; new=root/'unit'; backup=admin/'before'; backup.mkdir()
            old.write_bytes(b'after'); new.write_bytes(b'new'); (backup/'entry').write_bytes(b'before')
            rows={str(old):dict(after_sha256=common.sha(old),before_sha256=common.sha(backup/'entry'),backup='before/entry',before_mode=0o755,before_uid=0,before_gid=0),
                  str(new):dict(after_sha256=common.sha(new),before_sha256=None)}
            common.write(admin/'installation.json',dict(state='installed',targets=rows))
            with patch.object(install,'ADMIN',admin),patch.object(install,'idle'),patch.object(install,'command'),patch.object(install.os,'chown'):
                new.write_bytes(b'somebody changed it')
                with self.assertRaises(RuntimeError): install.restore()
                self.assertEqual(old.read_bytes(),b'after')
                new.write_bytes(b'new'); install.restore()
                self.assertEqual(old.read_bytes(),b'before'); self.assertFalse(new.exists())
    def test_historical_acceptance_cannot_open_current_source(self):
        import enable
        p=common.ROOT/'results/unified_experiments_20261011_v2/UKS/runs/20261011-155228-1471501'
        if not p.exists(): self.skipTest('Historical author result is not part of installed package')
        with patch.object(enable,'sha',return_value='different-version'):
            with self.assertRaisesRegex(RuntimeError,'different entry version'):
                enable.acceptance('UKS',p)

class Runner(unittest.TestCase):
    def exercise(self, code=0, source_changed=False):
        import runner
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp)/'root'; root.mkdir()
            admin=Path(tmp)/'admin'; admin.mkdir()
            out=Path(tmp)/'public'
            common.write(admin/'registry.json',dict(datasets={'UKS':dict(enabled=True,binding='binding.json',acceptance={'test_fixture':True})}))
            common.write(admin/'binding.json',dict(entry_sha256='bound-source'))
            for key,value in [('ROOT',root),('ADMIN',admin),('OUT',out)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(patch.object(runner.os,'geteuid',return_value=0))
            stack.enter_context(patch.object(runner.sys,'argv',['runner.py','UKS','short1']))
            stack.enter_context(patch.dict(runner.os.environ,{'INVOCATION_ID':'1'*32}))
            verify=stack.enter_context(patch.object(runner,'verify_binding'))
            if source_changed:verify.side_effect=[None,RuntimeError('source changed after worker')]
            old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
            for s,h in old.items():stack.callback(signal.signal,s,h)
            def launch(cmd,**kwargs):
                work=Path(cmd[-1]); common.write(work/'status.json',dict(stage='complete',passed=True,complete=True,source_sha256='bound-source'))
                common.write(work/'summary.json',result())
                common.write(work/'config.json',{})
                common.write(work/'backend.json',dict(output=str(root/'results/backend')))
                (work/'summary.csv').write_text('dataset,mode\nUKS,short1\n')
                process=Mock(returncode=code); process.poll.return_value=code
                return process
            stack.enter_context(patch.object(runner.subprocess,'Popen',side_effect=launch))
            if code or source_changed:
                with self.assertRaises(RuntimeError):runner.main()
            else:runner.main()
            latest=common.read(out/'UKS/short1/latest.json'); folder=Path(latest['output'])
            if code or source_changed:
                self.assertTrue((folder/'error.json').is_file()); self.assertFalse((folder/'completion.json').exists())
            else:
                self.assertEqual(common.read(folder/'completion.json')['summary_sha256'],common.sha(folder/'summary.json'))
                self.assertTrue((folder/'summary.csv').is_file())
    def test_success_publishes_complete_evidence(self):self.exercise()
    def test_failed_exit_never_publishes_pass(self):self.exercise(code=1)
    def test_source_changed_after_execution_never_publishes_pass(self):self.exercise(source_changed=True)

if __name__=='__main__':
    unittest.main(verbosity=2)
