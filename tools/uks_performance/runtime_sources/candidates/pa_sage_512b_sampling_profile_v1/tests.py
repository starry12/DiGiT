"""Test timing accounting, unmodified CUDA call body and report finalization."""
import ast
import copy
import tempfile
import unittest
from types import SimpleNamespace
from .common import *
from .profile import HostProfile, install, off_summary
from .analyze import analyze, breakdown, markdown


class StripScopes(ast.NodeTransformer):
    def visit_With(self,node):
        self.generic_visit(node)
        return node.body


class ProfileTests(unittest.TestCase):
    def test_nested_exclusive_and_exception_accounting(self):
        ticks=iter([0,10,30,50,60,80]);p=HostProfile(lambda:next(ticks))
        with p.span('outer'):
            with p.span('inner'):pass
        with self.assertRaises(ValueError):
            with p.span('failed'):raise ValueError('expected')
        s=p.summary()['spans']
        self.assertAlmostEqual(s['outer']['inclusive_seconds'],50e-9)
        self.assertAlmostEqual(s['outer']['exclusive_seconds'],30e-9)
        self.assertAlmostEqual(s['outer/inner']['inclusive_seconds'],20e-9)
        self.assertAlmostEqual(s['failed']['inclusive_seconds'],20e-9)
        self.assertEqual(p.stack,[])

    def test_scopes_are_only_change_to_native_frontier(self):
        paths=[ROOT/'candidates/pa_sage_bidir_native_v2/runtime/digit/sampler.py',HERE/'sampler.py']
        methods=[]
        for path in paths:
            tree=ast.parse(path.read_text())
            method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_cuda_group_frontier')
            methods.append(ast.dump(StripScopes().visit(method),include_attributes=False))
        self.assertEqual(*methods)

    def test_restore_instance_method_and_return_exception(self):
        class Target:
            def value(self,x):
                if x<0:raise ValueError('negative')
                return x+1
        p=HostProfile();t=Target();p.wrap(t,'value','value')
        self.assertEqual(t.value(4),5)
        with self.assertRaises(ValueError):t.value(-1)
        p.close();self.assertNotIn('value',vars(t));self.assertEqual(t.value(4),5)
        self.assertEqual(p.summary()['spans']['value']['calls'],2)

    def test_no_instrumentation_device_synchronization(self):
        for name in ('profile.py','sampler.py'):
            tree=ast.parse((HERE/name).read_text())
            calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            self.assertNotIn('synchronize',calls);self.assertNotIn('Event',calls)

    def test_balanced_full_schedule(self):
        for arm in ('gids','digit'):
            self.assertEqual([m for a,m in full_schedule() if a==arm],['off','host','host','off'])

    def test_cpu_dgl_real_hooks_and_restore(self):
        setup()
        import torch
        import dgl
        g=dgl.graph(([0,1,2,3,0,2],[1,2,3,0,2,0]),num_nodes=4)
        sampler=dgl.dataloading.NeighborSampler([-1,-1,-1],fused=False)
        torch.manual_seed(0);dgl.seed(0);dgl.random.seed(0)
        baseline_result=sampler.sample(g,torch.tensor([0,2]))
        loader=SimpleNamespace(resolve_batch_feature_rows=lambda b:b[0],window_buffering=lambda b:None,_read_many=lambda b:b)
        p=HostProfile();install(p,sampler,g,loader,'gids')
        torch.manual_seed(0);dgl.seed(0);dgl.random.seed(0)
        result=sampler.sample(g,torch.tensor([0,2]));p.close()
        self.assertTrue(torch.equal(result[0],baseline_result[0]))
        for a,b in zip(result[2],baseline_result[2]):
            self.assertTrue(torch.equal(a.edata[dgl.EID],b.edata[dgl.EID]))
        spans=p.summary()['spans']
        self.assertEqual(spans['sampler']['calls'],1)
        for layer in range(3):
            self.assertEqual(spans['sampler/blocks/layer%d.to_block'%layer]['calls'],1)
        self.assertNotIn('sample_neighbors',vars(g));self.assertNotIn('sample',vars(sampler))


def fixture_report(arm,mode):
    # Use an actually accepted v4 report to exercise inherited acceptance/footer.
    folder=Path(read(baseline.OUT/'status.json')['directory'])
    r=read(folder/('full_'+arm)/'report.json')
    if mode=='off':
        r['sampling_profile']=off_summary();return r
    n=r['updates'];spans={}
    def add(key,calls,inc,exc=None):
        spans[key]=dict(calls=calls,inclusive_seconds=inc,exclusive_seconds=inc if exc is None else exc)
    leaves=[]
    for layer in range(3):
        leaves.append(('sampler/blocks/'+('block_build/' if arm=='digit' else '')+'layer%d.to_block'%layer,n))
        if layer or arm=='gids':leaves.append(('sampler/blocks/layer%d.dgl_neighbors'%layer,n))
    if arm=='digit':
        for k in ('metadata_lookup','output_allocation','native_call','valid_nonzero','compact_indices','frontier_graph','edge_annotations'):
            leaves.append(('sampler/blocks/outer/'+k,n))
        leaves.append(('sampler/blocks/storage_annotation',n))
    for key,calls in leaves:add(key,calls,.1)
    total=len(leaves)*.1
    add('sampler',n,total,0);add('sampler/blocks',n,total,0)
    add('window_hint/feature_row_resolution',n,.01)
    add('merged_read/feature_row_resolution',n,.01)
    r['sampling_profile']=dict(off_summary(),mode='host',spans=spans)
    return r


class AcceptanceTests(unittest.TestCase):
    def test_footer_validates_and_analysis_reconciles(self):
        reports=[fixture_report(a,m) for a,m in full_schedule()]
        result=analyze(reports)
        self.assertTrue(result['passed']);self.assertIn('block',markdown(result))
        for r in reports:
            if r['sampling_profile']['mode']=='host':
                self.assertAlmostEqual(sum(breakdown(r).values()),r['timing']['sample_host_seconds'])

    def test_incomplete_counts_and_address_resolution_rejected(self):
        r=fixture_report('digit','host')
        for key in ('sampler/blocks/outer/native_call','sampler/blocks/block_build/layer0.to_block',
                    'merged_read/feature_row_resolution'):
            bad=copy.deepcopy(r);bad['sampling_profile']['spans'][key]['calls']-=1
            with self.assertRaises(RuntimeError):validate_completion(bad)

    def test_changed_workload_and_incomplete_schedule_rejected(self):
        reports=[fixture_report(a,m) for a,m in full_schedule()]
        with self.assertRaises(RuntimeError):analyze(reports[:-1])
        reports[2]['shapes'][0]['input_nodes']+=1
        with self.assertRaises(RuntimeError):analyze(reports)

    def test_off_must_be_uninstrumented(self):
        r=fixture_report('gids','off');validate_completion(r)
        r['sampling_profile']['spans']['unexpected']={}
        with self.assertRaises(RuntimeError):validate_completion(r)


if __name__=='__main__':
    setup()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    write(OUT/'cpu_checks.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),gpu_used=False,raw_ssd_used=False,
        tested_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_files()}))
    sys.exit(0 if result.wasSuccessful() else 1)
