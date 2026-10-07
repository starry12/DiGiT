"""CPU-only checks: routing, reference identity and final-result presentation."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('reviewer_cli', Path(__file__).with_name('cli.py'))
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

class ReviewerCLI(unittest.TestCase):
    def setUp(self):
        mock = patch.object(Path, 'is_file', return_value=True)
        mock.start()
        self.addCleanup(mock.stop)

    def test_routes(self):
        cases = [('run PA sage','main'),('run PA gcn','main'),('run PA gat','main'),
                 ('ablation PA sage','ablation'),('layout PA sage','layout'),
                 ('performance IG sage','IG'),('performance UKS sage','UKS'),
                 ('status PA sage','main'),('stop PA sage --action ablation','ablation')]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(cli.parse(args.split()).route,expected)
                route = cli.HANDLERS[expected]
                self.assertNotIn('legacy_cli',str(route))
        for dataset in ('IG','UKS','UKL','CL'):
            for command in cli.INSPECT:
                self.assertEqual(cli.parse(f'{command} {dataset} sage --action performance'.split()).route,dataset)

    def test_new_model_routes_and_reference_isolation(self):
        for dataset in ('IG','UKS','UKL','CL'):
            for model in ('gcn','gat'):
                for command in ('performance', *cli.INSPECT):
                    argv = [command,dataset,model] + ([] if command == 'performance' else ['--action','performance'])
                    self.assertEqual(cli.parse(argv).route,dataset+'_'+model)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    cli.parse(['results',dataset,model,'--action','performance','--reference'])

    def test_rejects_unpublished_or_ambiguous_workloads(self):
        cases=['run IG sage','run UKL sage','performance PA sage','performance UKS other',
               'layout PA gat','run PA sage --action run','results PA sage --reference',
               'status IG sage','performance IG sage --json','stop UKS sage --action performance --reference']
        for args in cases:
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                cli.parse(args.split())

    def test_main_accuracy_and_ratio_only(self):
        a=cli.parse('results PA sage --action run'.split());stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli.display(a,dict(state='PASS',results=[dict(ratios=dict(training_speedup=1.8027),rows=[
                dict(arm='gids',test_accuracy=.6285,training_seconds=500),
                dict(arm='digit_full',test_accuracy=.6296,training_seconds=280)])]))
        value=stream.getvalue()
        self.assertIn('1.80×',value);self.assertIn('62.85%',value);self.assertIn('62.96%',value)
        self.assertNotIn('500',value);self.assertNotIn('280',value)

    def test_unaccepted_results_not_shown(self):
        for state in ('FAILED','INCOMPLETE','RUNNING','AUTHOR_REFERENCE'):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                cli.display(cli.parse('results UKS sage --action performance'.split()),dict(state=state,summary=dict(speedup=99)))
            self.assertNotIn('99',stream.getvalue())

    def test_uks_accepted_reference(self):
        value=cli.uks_reference()
        self.assertEqual(value['state'],'AE_REFERENCE')
        self.assertEqual(cli.ratio(value['summary']['speedup']),'2.00×')

    def test_reference_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'reference').mkdir()
            name='reference/uks_sage_reviewer_acceptance.json'
            (root/name).write_text('{}')
            (root/'ARTIFACT_MANIFEST.json').write_text(json.dumps(dict(files={name:'0'*64})))
            with patch.object(cli,'ROOT',root),self.assertRaisesRegex(ValueError,'identity'):
                cli.uks_reference()

    def test_inspection_does_not_launch_or_change_state(self):
        a='results PA sage --action ablation'.split()
        response=type('Response',(),dict(returncode=0,stderr='',stdout=json.dumps(dict(
            state='PASS',summary=dict(arms=dict(digit_full=dict(speedup_vs_gids=1.877)))))))()
        with patch.object(cli.subprocess,'run',return_value=response) as run,patch.object(cli.os,'execv') as execute,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(a),0)
        execute.assert_not_called()
        self.assertEqual(run.call_args.args[0],['/usr/bin/python3','-I','-B',str(cli.HANDLERS['ablation']),*a,'--json'])

    def test_failed_handler_return_code_preserved(self):
        response=type('Response',(),dict(returncode=3,stderr='',stdout='{"state":"INCOMPLETE"}'))()
        with patch.object(cli.subprocess,'run',return_value=response),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main('results PA sage'.split()),3)

    def test_valid_launch_dispatches_once(self):
        with patch.object(cli.os,'execv') as execute:
            cli.main('performance IG sage'.split())
        execute.assert_called_once_with('/usr/bin/python3',['/usr/bin/python3','-I','-B',str(cli.HANDLERS['IG']),'performance','IG','sage'])

if __name__=='__main__':unittest.main()
