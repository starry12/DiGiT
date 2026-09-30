"""Counter semantics tests; --cuda runs the actual compiled device helpers."""
import argparse,copy,io,json,random,sys,unittest
from pathlib import Path
from candidates.io_accounting_v1.accounting import *
from candidates.io_accounting_v1.common import HERE,sha

def region(useful=1024,physical=4096,active_ns=1000000,cpu=2,gpu=3,width=512,replay=0):
    a=decode([1,1,0,0,0,0]);b=decode([1,1,physical-replay,useful,gpu*width,gpu])
    return dict(useful_io=useful_interval(a,b,dict(cpu=cpu,gpu_ssd=gpu),width),
                feature=dict(cpu=cpu,gpu_ssd=gpu),feature_row_bytes=width,feature_seconds=.01,
                device=dict(enabled=True,reconciled=True,primary_bytes=physical-replay,replay_bytes=replay,
                            completed_bytes=physical,active_ns=active_ns,submitted_commands=1,completed_commands=1))
class AccountingTests(unittest.TestCase):
    def test_padding_and_replay(self):
        r=region(physical=8192,replay=4096);s=summarize([(r,2.)])
        self.assertEqual(s['ssd_useful_bytes'],1024);self.assertEqual(s['ssd_payload_utilization'],.125)
        self.assertEqual(s['logical_feature_bytes'],2560)
    def test_ratio_of_sums(self):
        r=region();q=region(useful=512,physical=4096,active_ns=3000000)
        s=summarize([(r,1.),(q,3.)]);self.assertEqual(s['ssd_useful_gbps'],1536/.004/1e9)
    def test_warm_cache_and_cpu_only(self):
        r=region(useful=0,physical=0,active_ns=0,gpu=0,cpu=2,width=4096)
        r['device'].update(submitted_commands=0,completed_commands=0)
        s=summarize([(r,1.)]);self.assertIsNone(s['ssd_useful_gbps']);self.assertEqual(s['logical_feature_bytes'],8192)
    def test_reject_region_switch(self):
        with self.assertRaises(RuntimeError):useful_interval(decode([1,1,0,0,0,0]),decode([1,2,0,0,0,0]),dict(cpu=0,gpu_ssd=0))
    def test_reject_disabled(self):
        with self.assertRaises(RuntimeError):useful_interval(decode([0,0,0,0,0,0]),decode([0,0,0,0,0,0]),dict(cpu=0,gpu_ssd=0))
    def test_reject_incomplete_coverage(self):
        with self.assertRaises(RuntimeError):useful_interval(decode([1,1,0,0,0,0]),decode([1,1,4096,512,512,1]),dict(cpu=0,gpu_ssd=2))
    def test_reject_nonfinite_and_negative(self):
        for name in ('active_ns','primary_bytes','completed_bytes'):
            r=region();r['device'][name]=-1
            with self.assertRaises(RuntimeError):summarize([(r,1.)])
        with self.assertRaises(RuntimeError):summarize([(region(),float('nan'))])
        r=region();r['feature_seconds']=float('nan')
        with self.assertRaises(RuntimeError):summarize([(r,1.)])
    def test_reject_fill_and_completion_mismatch(self):
        for field in ('ssd_fill_bytes','ssd_useful_bytes'):
            r=region();r['useful_io'][field]+=4096
            with self.assertRaises(RuntimeError):validate_region(r)
        r=region();r['device']['completed_commands']=0
        with self.assertRaises(RuntimeError):validate_region(r)
    def test_window_carry_is_not_a_full_region(self):
        a=decode([1,1,4096,512,512,1]);b=decode([1,1,4096,1024,1024,2])
        d=useful_interval(a,b,dict(cpu=0,gpu_ssd=1));self.assertEqual(d['ssd_fill_bytes'],0)
        self.assertEqual(d['ssd_useful_bytes'],512)

def reference(events,slots):
    masks=[(1<<64)-1]*slots;values=[1,1,0,0,0,0];out=[]
    for op,slot,offset,bytes_ in events:
        bits=((1<<(bytes_//512))-1)<<(offset//512)
        if op==0:masks=[(1<<64)-1]*slots;values=[1,values[1]+1,0,0,0,0]
        elif op==5:values[0]=0
        elif values[0]:
            if op in (1,2):
                masks[slot]=((1<<64)-1)&~bits if op==1 else masks[slot]&~bits
                values[2]+=bytes_
            else:
                values[3]+=bin(bits&~masks[slot]).count('1')*512;masks[slot]|=bits
                repeats=1024 if op==4 else 1;values[4]+=bytes_*repeats;values[5]+=repeats
        out.extend(values)
    return out

def cuda_tests():
    sys.path.insert(0,str(HERE/'runtime'))
    import BAM_Feature_Store as native
    import torch
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==(HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so').resolve(),'Wrong native module')
    cases={
      'pa_g2_padding':([[1,0,0,4096],[3,0,0,512],[3,0,512,512],[3,0,0,512]],4096),
      'concurrent_repeat_1024':([[1,0,0,4096],[4,0,0,512]],4096),
      'eviction_refill':([[1,0,0,4096],[3,0,0,512],[1,0,0,4096],[3,0,0,512]],4096),
      'warm_region_excluded':([[1,0,0,4096],[0,0,0,512],[3,0,0,512]],4096),
      'partial_after_boundary':([[1,0,0,512],[0,0,0,512],[2,0,512,512],[3,0,0,512],[3,0,512,512]],4096),
      'ig_8k_two_rows':([[1,0,0,8192],[3,0,0,4096],[3,0,4096,4096],[3,0,0,4096]],8192),
      'ig_split_entries':([[1,0,0,4096],[1,1,0,4096],[3,0,0,4096],[3,1,0,4096]],4096),
      'disabled':([[5,0,0,512],[1,0,0,4096],[4,0,0,512]],4096),
      'independent_slots':([[1,0,0,4096],[1,1,0,4096],[3,0,0,512],[3,1,0,512]],4096),
    }
    rng=random.Random(13)
    cases['randomized_200_events']=([[rng.choice([0,1,2,3,4]),rng.randrange(2),rng.randrange(8)*512,512] for _ in range(200)],4096)
    evidence={}
    for name,(events,page) in cases.items():
        observed=list(native.useful_counter_probe(events,2,page));expected=reference(events,2)
        require(observed==expected,'CUDA counter mismatch: '+name)
        evidence[name]=dict(passed=True,events=len(events),final_counters=decode(observed[-6:]))
    for bad in ([[3,0,4096,512]],[[3,0,0,128]],[[1,9,0,512]]):
        try:native.useful_counter_probe(bad,2,4096)
        except ValueError:pass
        else:raise RuntimeError('CUDA probe accepted bad geometry')
    # Independent exact expected values, beyond the reference interpreter.
    require(evidence['pa_g2_padding']['final_counters']['ssd_useful_bytes']==1024,'g2 padding counted')
    require(evidence['concurrent_repeat_1024']['final_counters']['ssd_useful_bytes']==512,'Concurrent double counting')
    require(evidence['warm_region_excluded']['final_counters']['ssd_useful_bytes']==0,'Cross-region credit')
    return dict(passed=True,gpu=torch.cuda.get_device_name(0),cases=evidence,
                native_sha256=sha(HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'),
                raw_ssd_access=False,scope='real CUDA counter primitives, not native SSD/feature-copy acceptance')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');parser.add_argument('--output',type=Path);a=parser.parse_args()
    stream=io.StringIO();run=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AccountingTests))
    print(stream.getvalue());result=dict(passed=run.wasSuccessful(),cpu_tests=run.testsRun,cpu_log=stream.getvalue())
    if a.cuda and result['passed']:result['cuda']=cuda_tests()
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
if __name__=='__main__':main()
