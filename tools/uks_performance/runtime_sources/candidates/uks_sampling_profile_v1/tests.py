import ast,unittest
from pathlib import Path
from .common import HERE,ROOT
from .host_profile import HostProfile
class Strip(ast.NodeTransformer):
    def visit_With(self,n):self.generic_visit(n);return n.body
class Tests(unittest.TestCase):
    def test_frontier_only_adds_scopes(self):
        bodies=[]
        for p in [ROOT/'candidates/pa_sage_bidir_native_v2/runtime/digit/sampler.py',HERE/'sampler.py']:
            tree=ast.parse(p.read_text());f=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_cuda_group_frontier');bodies.append(ast.dump(Strip().visit(f),include_attributes=False))
        self.assertEqual(*bodies)
    def test_kernel_bodies_unchanged(self):
        def body(s,name):
            start=s.index('__global__ void '+name);i=s.index('{',start)+1;n=1
            while n:n+=(s[i]=='{')-(s[i]=='}');i+=1
            return s[start:i]
        old=(ROOT/'candidates/pa_sage_bidir_native_v2/native/digit_sampler_cuda.cu').read_text();new=(HERE/'native/sampler.cu').read_text()
        for n in ('group_sample_kernel','resolve_eids_kernel'):self.assertEqual(body(old,n),body(new,n))
        self.assertIn('PYBIND11_MODULE(UKSSamplingProbeCUDA, module)',new)
    def test_exclusive_not_double_counted(self):
        it=iter([0,10,30,50]);p=HostProfile(lambda:next(it))
        with p.span('outer'):
            with p.span('inner'):pass
        r=p.summary()['spans'];self.assertEqual(r['outer']['inclusive_seconds'],50e-9);self.assertEqual(r['outer']['exclusive_seconds'],30e-9)
    def test_no_cuda_sync_in_host_frontier(self):
        for name in ['host_profile.py','sampler.py']:
            tree=ast.parse((HERE/name).read_text())
            self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('synchronize','Event') for n in ast.walk(tree)))
    def test_restore_preserves_method(self):
        class T:
            def f(self):return 7
        p=HostProfile();t=T();p.wrap(t,'f','wrapped');self.assertEqual(t.f(),7);p.close();self.assertNotIn('f',vars(t));self.assertEqual(t.f(),7)
if __name__=='__main__':unittest.main()
