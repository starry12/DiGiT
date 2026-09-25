"""Bounded CPU tests of privileged command boundaries and acceptance decisions."""
import contextlib,copy,importlib.util,io,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
H=Path(__file__).resolve().parent
def module(name):
    spec=importlib.util.spec_from_file_location('layout_test_'+name,H/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
cli=module('cli');controller=module('controller')

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ae-layout-cpu-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.control=self.root/'control';self.outputs=self.root/'outputs'
        (self.control/'state').mkdir(parents=True);self.outputs.mkdir()
        for name,val in [('CONTROL',self.control),('OUTPUTS',self.outputs)]:
            p=patch.object(cli,name,val);p.start();self.addCleanup(p.stop)

    def write(self,p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))

    def build(self):
        out=self.outputs/'new';out.mkdir()
        rows=[]
        for point in sorted(cli.POINTS):
            f=out/'native'/point/'full_digit_full_accepted.json';self.write(f,dict(passed=True,point=point))
            rows.append(dict(point=dict(id=point),accepted_report_sha256=cli.sha(f)))
        self.write(out/'summary.json',dict(passed=True,points=rows))
        s=dict(passed=True,complete=True,stage='complete',native_acceptance=True,invocation_id='new',
               summary_sha256=cli.sha(out/'summary.json'),steps=[],completed=sorted(cli.POINTS))
        self.write(out/'status.json',s);self.write(self.control/'state/latest.json',dict(output=str(out)))
        return out,s

    def service(self,active='inactive',result='success',code='0',invocation='new'):
        return patch.object(cli.subprocess,'run',return_value=SimpleNamespace(stdout=
            'ActiveState='+active+'\nResult='+result+'\nExecMainStatus='+code+'\nInvocationID='+invocation+'\n'))

    def test_cli_fixed_scope_and_no_arbitrary_paths(self):
        self.assertEqual(cli.parse(['layout','PA','sage']).action,'layout')
        for args in (['layout','IG','sage'],['layout','PA','sage','--output','/tmp/a'],['stop','PA','sage'],
                     ['layout','PA','sage','--reference'],['layout','PA','sage','--action','layout']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.parse(args)

    def test_start_fixed_service_and_no_extra_arguments(self):
        with patch.object(cli.subprocess,'run') as r,contextlib.redirect_stdout(io.StringIO()):cli.main(['layout','PA','sage'])
        self.assertEqual(r.call_args.args[0],['/usr/bin/sudo','-n','/usr/bin/systemctl','--no-block','start',cli.UNIT])

    def test_reference_is_explicit_and_does_not_launch(self):
        self.write(self.control/'author_reference.json',dict(points=[],passed=True))
        with patch.object(cli.subprocess,'run') as r,contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(['results','PA','sage','--action','layout','--reference','--json']),0)
        self.assertEqual(json.loads(out.getvalue())['state'],'AUTHOR_REFERENCE');r.assert_not_called()

    def test_not_started_does_not_substitute_author_result(self):
        self.write(self.control/'author_reference.json',dict(passed=True))
        self.assertEqual(cli.view()['state'],'NOT_STARTED')

    def test_path_traversal_and_symlinks(self):
        out=self.outputs/'one';out.mkdir();link=self.outputs/'link';link.symlink_to(out)
        for p in (link,self.outputs/'..'/'escape',Path('/tmp/unrelated')):
            with self.assertRaises(ValueError):cli.checked_output(str(p))
        with self.assertRaises(ValueError):cli.checked_file(out,'../status.json')
        (out/'link').symlink_to(self.root)
        with self.assertRaises(ValueError):cli.checked_file(out,'link/data.json')

    def test_pass_requires_service_success_and_invocation(self):
        self.build()
        for kw,state in [({},'PASS'),({'active':'active'},'FINALIZING'),({'code':'1'},'FAILED'),
                         ({'result':'exit-code'},'FAILED'),({'invocation':'old'},'FAILED')]:
            with self.subTest(kw=kw),self.service(**kw):self.assertEqual(cli.view()['state'],state)

    def test_failed_latest_is_not_previous_pass(self):
        out,s=self.build();s.update(passed=False,complete=False,stage='failed');self.write(out/'status.json',s)
        with self.service():self.assertEqual(cli.view()['state'],'FAILED')

    def test_final_report_drift_rejected(self):
        out,s=self.build();self.write(out/'native/g1_r00/full_digit_full_accepted.json',dict(changed=True))
        with self.service(),self.assertRaises(ValueError):cli.view()

    def test_incomplete_grid_and_summary_drift_rejected(self):
        out,s=self.build();m=json.loads((out/'summary.json').read_text());m['points'].pop();self.write(out/'summary.json',m)
        with self.service(),self.assertRaises(ValueError):cli.view()
        s['summary_sha256']=cli.sha(out/'summary.json');self.write(out/'status.json',s)
        with self.service(),self.assertRaises(ValueError):cli.view()

    def point(self,key='g1_r00'):
        out=self.root/key;out.mkdir();modes=['smoke','full'] if key=='g2_r20' else ['full']
        workers=[dict(mode=m,arm='digit_full',returncode=0) for m in modes]
        s=dict(passed=True,complete=True,stage='complete',point=dict(id=key),workers=workers,candidate_sha256=controller.NATIVE_SHA)
        r=dict(passed=True,smoke=False,source_only=False,updates=1179,epochs=[{}],test=None,point=dict(id=key),candidate_sha256=controller.NATIVE_SHA)
        self.write(out/'status.json',s);self.write(out/'full_digit_full_accepted.json',r)
        self.write(out/'full_summary.json',dict(passed=True,report_sha256=dict(digit_full=controller.sha(out/'full_digit_full_accepted.json'))))
        for mode in modes:self.write(out/(mode+'_monitor_digit_full')/'external_gpu/summary.json',dict(passed=True,complete=True,errors=[]))
        return out,s,r

    def test_point_requires_full_epoch_and_native_identity(self):
        out,s,r=self.point();controller.validate_point(out,'g1_r00')
        for changes in (dict(updates=1178),dict(source_only=True),dict(candidate_sha256='different')):
            self.write(out/'full_digit_full_accepted.json',dict(r,**changes))
            with self.assertRaises(RuntimeError):controller.validate_point(out,'g1_r00')

    def test_strict_pilot_requires_smoke(self):
        out,s,r=self.point('g2_r20');controller.validate_point(out,'g2_r20')
        s['workers']=s['workers'][1:];self.write(out/'status.json',s)
        with self.assertRaises(RuntimeError):controller.validate_point(out,'g2_r20')

    def test_monitor_error_blocks_acceptance(self):
        out,s,r=self.point();self.write(out/'full_monitor_digit_full/external_gpu/summary.json',dict(passed=True,complete=True,errors=['timeout']))
        with self.assertRaises(RuntimeError):controller.validate_point(out,'g1_r00')

    def test_point_runner_keeps_exact_native_flow_except_locks(self):
        # Native source lives in the sibling private snapshot for deployment tests.
        source=H.parent/'snapshot/candidates/pa_sage_layout_shared_resume_v3/run.py'
        if not source.exists():source=H/'runtime_sources/candidates/pa_sage_layout_shared_resume_v3/run.py'
        if not source.exists():self.skipTest('Private deployment source is not distributed with public CPU tests')
        text=source.read_text();lock="    global_lock=open('/run/digit-ae-selfservice/exclusive.lock','a+')\n    fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    lock=open('/tmp/digit-pa-bidir-controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        text=text.replace(lock,"    require(str(os.getppid())==os.environ.get('DIGIT_LAYOUT_CONTROLLER_PID'),'Grid controller must own the shared locks')\n")
        text=text.replace('from candidates.pa_sage_layout_shared_resume_v3.common import *',"sys.path.insert(0,'/home/embed/digit')\nfrom candidates.pa_sage_layout_shared_resume_v3.common import *",1)
        self.assertEqual(text,(H/'point_runner.py').read_text())

if __name__=='__main__':
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))});os.nice(19-os.nice(0))
    unittest.main(verbosity=2)
