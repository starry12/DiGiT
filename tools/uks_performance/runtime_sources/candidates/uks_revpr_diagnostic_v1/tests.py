import unittest
from .timing import StageTimer
from .statistics import jobs,summarize,STAGES

def rows():
    result=[]
    for j in jobs():
        sec={'gids_default':12.,'digit_default':10.,'digit_cpu2':15.}[j['variant']]
        if j['profile_mode']=='stages':sec=300.
        windows=[{}]+[dict(stages={k:dict(seconds=1.,calls=100) for k in STAGES}) for _ in range(3)]
        result.append(dict(j,passed=True,normal_exit=True,updates=320,measured_batches=300,warmup_batches=20,windows_seconds=[sec/3]*3,seconds=sec,roots_sha256='r',initial_model_sha256='m',hot_file_sha256='h',cache={'logical_hot_sha256':'same'},windows=windows))
    return result
class Tests(unittest.TestCase):
    def test_excludes_instrumented_times(self):
        s=summarize(rows());self.assertEqual(s['comparisons']['digit_default']['max_observed_speedup'],1.2);self.assertEqual(s['comparisons']['digit_cpu2']['max_observed_speedup'],.8);self.assertEqual(s['profiles']['digit_cpu2']['stages']['backward']['calls'],300)
    def test_requires_same_hot_roots_and_complete_profile(self):
        for key in ('hot_file_sha256','roots_sha256'):
            r=rows();r[0][key]='changed'
            with self.assertRaises(ValueError):summarize(r)
        r=rows();r[0]['cache']['logical_hot_sha256']='different'
        with self.assertRaises(ValueError):summarize(r)
        r=rows();r[-1]['windows'][1]['stages']['sampling']['calls']=99
        with self.assertRaises(ValueError):summarize(r)
        with self.assertRaises(ValueError):summarize(rows()[:-1])
    def test_timer_disabled_no_clock_or_sync(self):
        def fail():raise AssertionError('Instrumentation in performance mode')
        t=StageTimer(fail,False,fail)
        with t.stage('sample'):pass
        self.assertEqual(t.report(),{})
    def test_timer_measures_fence_and_counts(self):
        calls=[];clock=iter([1.,1.25]);t=StageTimer(lambda:calls.append(1),True,lambda:next(clock))
        with t.stage('sample'):pass
        self.assertEqual(len(calls),2);self.assertEqual(t.report()['sample'],dict(seconds=.25,calls=1))
    def test_timer_rejects_double_counting(self):
        t=StageTimer(lambda:None,True)
        with self.assertRaises(ValueError):
            with t.stage('fetch'):
                with t.stage('sample'):pass
    def test_same_revpr_installer_cpu(self):
        import numpy as np
        from candidates.uks_native_v1.mapping import install
        class Fake:
            def capabilities(self):return dict(exact_logical_cpu_rows=True,replica_aliases=True,padding_uncached=True)
            def begin_exact_cpu_cache(self,rows,extent,row_bytes):self.rows=rows.copy()
            def write_cpu_row_map(self,lo,slots):pass
            def finish_exact_cpu_cache(self):pass
            def configure_gpu_cache(self,*args):pass
        hot=np.array([1,3],dtype=np.int64);original=np.arange(4,dtype=np.int64);storage=np.array([3,1,-1,-1,0,2,-1,-1,1,3,-1,-1],dtype=np.int64);primary=np.array([4,1,5,0],dtype=np.int64)
        a=Fake();b=Fake();budget=dict(cpu_rows=2,gpu_feature_cache_bytes=4*2**30,gpu_policy='legacy')
        x=install(a,budget,hot,original,original);y=install(b,dict(budget,gpu_policy='fifo'),hot,primary,storage)
        self.assertEqual(x['logical_hot_sha256'],y['logical_hot_sha256']);np.testing.assert_array_equal(storage[b.rows],hot)
if __name__=='__main__':unittest.main()
