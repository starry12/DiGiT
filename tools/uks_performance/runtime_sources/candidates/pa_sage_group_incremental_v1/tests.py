"""CPU controls: native binding, exact workload and trace/performance boundaries."""
import tempfile
import unittest
from .common import *
from .build import preserved_kernels
from .control import stages
from .analyze import analyze,compare_workload,markdown
from .trace import kernel_evidence,compare_traces
from .native_tests import fixture as graph_fixture, oracle


def fixture(mode):
    folder=Path(read(reference.OUT/'status.json')['directory'])
    r=read(folder/'full_02_graph/report.json')
    r.update(variant=mode,group_mode=mode,incremental_group_api=1,sampler_binary_sha256=sha(SAMPLER_BINARY))
    return r


class ProtocolTests(unittest.TestCase):
    def test_schedule_and_preserved_kernels(self):
        plan=stages()
        self.assertEqual([v for n,s,v in plan if not s and not n.startswith('trace_')],full_schedule())
        self.assertEqual([v for n,s,v in plan if s],['legacy','incremental'])
        self.assertEqual([n for n,s,v in plan if n.startswith('trace_')],['trace_legacy','trace_incremental'])
        self.assertEqual(read(HERE/'protocol.json'),read(reference.HERE/'protocol.json'))
        kernels=preserved_kernels()
        self.assertTrue(kernels['group_kernel_byte_identical'])
        self.assertTrue(kernels['legacy_eid_kernel_byte_identical'])

    def test_complete_summary(self):
        r=analyze([fixture(v) for v in full_schedule()])
        self.assertTrue(r['complete']);self.assertEqual(r['speedup'],1.)
        self.assertIn('incremental',markdown(r))

    def test_wrong_sampler_dense_and_io_modes(self):
        for key,value in [('group_mode','legacy'),('incremental_group_api',0),('sampler_binary_sha256','bad'),
                ('dense_mode','eager'),('eid_mode','shared'),('io_mode','legacy')]:
            r=fixture('incremental');r[key]=value
            with self.assertRaises(RuntimeError):validate_completion(r)
        r=fixture('incremental');r['dense_graph']['graph_forwards']=1177
        with self.assertRaises(RuntimeError):validate_completion(r)

    def test_workload_loss_profile_and_pending_reads(self):
        a=fixture('legacy')
        for key,value in [('updates',1178),('root_order_sha256','bad'),('initial_model_sha256','bad')]:
            r=fixture('incremental');r[key]=value
            with self.assertRaises(RuntimeError):compare_workload(a,r)
        r=fixture('incremental');r['losses'][0]+=.1
        with self.assertRaises(RuntimeError):compare_workload(a,r)
        r=fixture('incremental');r['async_backend']['outstanding']=1
        with self.assertRaises(RuntimeError):validate_completion(r)
        reports=[fixture(v) for v in full_schedule()];reports[0]['diagnostic_trace']=True
        with self.assertRaises(RuntimeError):analyze(reports)
        with self.assertRaises(RuntimeError):analyze([fixture('legacy'),fixture('incremental')])

    def test_cpu_oracle_preserves_occurrences_and_padding(self):
        f=graph_fixture(8,2,'groups')
        result=oracle(f,5,0)
        # Owners 1/2/7 have enough groups, but one remaining slot cannot fit a group.
        for pos in (1,2,7):
            self.assertEqual(result[4][pos],2)
            self.assertEqual(result[5][pos],4)
            self.assertEqual(result[0][pos*5+4],-1)
        f=graph_fixture(65,2,'normal');result=oracle(f,128,0)
        self.assertEqual(result[5][1],65)  # Only eight IDs; all 65 occurrences remain selectable.
        self.assertEqual(len(set(result[0][128:128+65])),8)

    def test_trace_requires_new_group_and_same_dense_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'trace.json';traces={}
            for mode in VARIANTS:
                name='group_sample_incremental_kernel<int>' if mode=='incremental' else 'group_sample_kernel<int>'
                events=[dict(ph='X',cat='kernel',name=name,dur=200),dict(ph='X',cat='kernel',name='resolve_eids_kernel<long>',dur=100)]*16
                events += [dict(ph='X',cat='cuda_runtime',name='cudaGraphLaunch',dur=2)]*32
                write(p,dict(traceEvents=events));traces[mode]=kernel_evidence(p,mode)
                with self.assertRaises(RuntimeError):kernel_evidence(p,'incremental' if mode=='legacy' else 'legacy')
            self.assertEqual(compare_traces(traces['legacy'],traces['incremental'])['group_kernel_reduction_percent'],0.)
            write(p,dict(traceEvents=events[:-1]))
            with self.assertRaises(RuntimeError):kernel_evidence(p,'incremental')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    write(OUT/'cpu_checks.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),gpu_used=False,raw_ssd_used=False,
        tested_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files()}))
    sys.exit(0 if result.wasSuccessful() else 1)
