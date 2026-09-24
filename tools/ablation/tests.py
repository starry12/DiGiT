"""CPU tests for fixed command routing, output trust and closed-result semantics."""
import contextlib, hashlib, importlib.util, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ablation_cli', H / 'cli.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)

class CliTests(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(cli.parse(['ablation', 'PA', 'sage']).action, 'ablation')
        for a in ('status', 'results', 'logs', 'stop'):
            self.assertEqual(cli.parse([a, 'PA', 'sage', '--action', 'ablation']).selector, 'ablation')

    def test_reject_extra_privileged_arguments(self):
        for args in [['ablation','IG','sage'], ['ablation','PA','gcn'], ['ablation','PA','sage','--gpu','1'],
                     ['ablation','PA','sage','--output','/tmp/x'], ['ablation','PA','sage','ssh.service'], ['stop','PA','sage']]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): cli.parse(args)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.control = self.root / 'control'; self.control.mkdir()
        self.outputs = self.root / 'results'; self.outputs.mkdir(); (self.control / 'state').mkdir()
        self.patches = [patch.object(cli, 'CONTROL', self.control), patch.object(cli, 'OUTPUTS', self.outputs)]
        for p in self.patches: p.start(); self.addCleanup(p.stop)

    def request(self, stage='complete', passed=True):
        folder = self.outputs / 'request'; folder.mkdir()
        (self.control / 'state/latest.json').write_text(json.dumps(dict(output=str(folder))))
        (folder / 'status.json').write_text(json.dumps(dict(stage=stage, passed=passed, complete=passed, workers=[])))
        return folder

    def service(self, active='inactive', result='success'):
        return patch.object(cli.subprocess, 'run', return_value=type('Result', (), dict(stdout='ActiveState='+active+'\nSubState=dead\nResult='+result+'\nExecMainStatus=0\n'))())

    def test_no_request_does_not_launch(self):
        with patch.object(cli.subprocess, 'run') as run:
            self.assertEqual(cli.view()['state'], 'NOT_STARTED'); run.assert_not_called()

    def test_reject_output_escape(self):
        for path in ['/tmp/anything', str(self.outputs / '..' / 'elsewhere')]:
            with self.assertRaises(ValueError): cli.checked_output(path)

    def test_reject_symlink_output(self):
        (self.outputs / 'escape').symlink_to(self.control, target_is_directory=True)
        with self.assertRaises(ValueError): cli.checked_output(self.outputs / 'escape')

    def test_no_pass_while_service_active(self):
        self.request()
        with self.service(active='active'): self.assertEqual(cli.view()['state'], 'FINALIZING')

    def test_new_failure_not_replaced_by_old_success(self):
        self.request(stage='failed', passed=False)
        with self.service(): self.assertEqual(cli.view()['state'], 'FAILED')

    def test_closed_summary_hash_required(self):
        folder = self.request(); (folder / 'full_summary.json').write_text(json.dumps(dict(passed=True, report_sha256={})))
        with self.service(), self.assertRaises(ValueError): cli.view()

    def test_accepted_report_hash_required(self):
        folder = self.request(); report = folder / 'full_gr_accepted.json'; report.write_text('{}')
        summary = folder / 'full_summary.json'; summary.write_text(json.dumps(dict(passed=True, report_sha256={'gr':cli.sha(report)})))
        state = json.loads((folder / 'status.json').read_text()); state['summary_sha256'] = cli.sha(summary)
        (folder / 'status.json').write_text(json.dumps(state))
        with self.service(): self.assertEqual(cli.view()['state'], 'PASS')
        report.write_text('{"changed":true}')
        with self.service(), self.assertRaises(ValueError): cli.view()

    def test_exact_sudo_start(self):
        with patch.object(cli.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            cli.main(['ablation','PA','sage'])
            run.assert_called_once_with(['/usr/bin/sudo','-n','/usr/bin/systemctl','--no-block','start',cli.UNIT], check=True)

if __name__ == '__main__': unittest.main()
