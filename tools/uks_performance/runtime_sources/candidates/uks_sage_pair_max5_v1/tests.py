import copy,unittest
from .statistics import jobs,summarize
from .common import load_plan

def fixture():
    pairs=[(20,10),(24,10),(45,30),(21,10),(22,10)]
    rows=[]
    for job in jobs():
        g,d=pairs[job['repetition']-1];t=g if job['arm']=='gids' else d
        rows.append(dict(job,passed=True,normal_exit=True,updates=320,warmup_batches=20,measured_batches=300,windows_seconds=[t/3]*3,seconds=t,roots_sha256='roots',initial_model_sha256='initial'))
    return rows
class Tests(unittest.TestCase):
    def test_paired_not_cross_round_max(self):
        s=summarize(fixture());self.assertEqual(s['selected_round'],2);self.assertEqual(s['max_observed_speedup'],2.4)
    def test_exact_tie_first(self):
        r=fixture()
        for x in r:x['seconds']=30 if x['arm']=='gids' else 15;x['windows_seconds']=[x['seconds']/3]*3
        self.assertEqual(summarize(r)['selected_round'],1)
    def test_missing_failed_unpaired_rejected(self):
        for bad in [fixture()[:-1],list(reversed(fixture()))]:
            with self.assertRaises(ValueError):summarize(bad)
        for key,bad in [('normal_exit',False),('measured_batches',299),('warmup_batches',0),('roots_sha256','changed'),('initial_model_sha256','changed'),('seconds',float('nan'))]:
            r=fixture();r[0][key]=bad
            with self.assertRaises(ValueError):summarize(r)
    def test_raw_precision_selection(self):
        r=fixture()
        for x in r:
            x['seconds']=10 if x['arm']=='digit' else (20.00001 if x['repetition']==5 else 20);x['windows_seconds']=[x['seconds']/3]*3
        self.assertEqual(summarize(r)['selected_round'],5)
    def test_reuse_plans_without_feature_reads(self):
        from unittest.mock import patch
        from candidates.uks_native_v1.storage import api
        with patch.object(api(),'build_plain_payload_plan',side_effect=AssertionError('No rehash/rewrite')):
            for arm,offset in [('gids',4*2**40),('digit',int(4.25*2**40))]:
                p=load_plan(arm);self.assertEqual(p.device_offset_bytes,offset);self.assertEqual(p.feature_dim,256)
    def test_window_and_round_contract(self):
        import ast
        from pathlib import Path
        from . import worker
        tree=ast.parse(Path(worker.__file__).read_text())
        self.assertEqual(len(jobs()),10)
        self.assertEqual([j['arm'] for j in jobs()],['gids','digit','digit','gids','gids','digit','digit','gids','gids','digit'])
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and any(x.name in ('prepare','verify_payload') for x in n.names) for n in ast.walk(tree)))
if __name__=='__main__':unittest.main()
