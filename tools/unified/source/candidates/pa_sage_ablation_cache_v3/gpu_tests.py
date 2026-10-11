"""Tiny CUDA row lookup and resident-probe tests; never open a raw device."""
import argparse,os,subprocess,time
from .common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    require(not a.output.exists(),'Preserve prior evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Only idle physical GPU 2')
    uuid=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    require(uuid not in apps,'GPU 2 has an active process')
    setup();import numpy as np
    import BAM_Feature_Store as bam
    from .cache import logical_slots,map_storage
    binary=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==binary,'Wrong native binary')
    began=time.time();cases=[]
    for seed in range(8):
        rng=np.random.RandomState(seed);n=31;hot=np.sort(rng.choice(n,11,replace=False))
        storage=rng.randint(-1,n,size=257)
        # Explicit same-page cold/hot neighbors, padding and replica on another page.
        storage[:10]=[hot[0],-1,next(x for x in range(n) if x not in hot),hot[1],hot[0],-1,hot[2],hot[0],hot[0],hot[1]]
        mapping=map_storage(storage,logical_slots(hot,n,np),np)
        features=np.arange(n*128,dtype=np.float32).reshape(n,128)
        requests=np.arange(len(storage)+2,dtype=np.uint64)
        actual=np.asarray(bam.exact_cpu_probe(mapping.tolist(),features[hot].reshape(-1).tolist(),requests.tolist(),128)).reshape(-1,128)
        expected=np.full_like(actual,-1)
        for i,row in enumerate(requests):
            if row<len(storage) and storage[row] in hot:expected[i]=features[storage[row]]
        np.testing.assert_array_equal(actual,expected)
        cases.append(dict(seed=seed,physical_rows=len(storage),logical_hot_rows=len(hot),requests=len(requests),passed=True))
    invalid=bam.exact_cpu_probe([0,1,3,4294967295],list(range(256)),[0,1,2,3,4],128)
    np.testing.assert_array_equal(np.asarray(invalid).reshape(-1,128)[[0,2,3,4]],-np.ones((4,128)))
    np.testing.assert_array_equal(np.asarray(invalid).reshape(-1,128)[1],np.arange(128))
    probe=np.asarray(bam.full_resident_probe_test(),dtype=np.uint32).reshape(6,3)
    np.testing.assert_array_equal(probe[:,0],[1,0,0,0,1,0])
    np.testing.assert_array_equal(probe[[0,4],1]-1,probe[[0,4],2])
    np.testing.assert_array_equal(probe[[1,2,3,5],1],probe[[1,2,3,5],2])
    write(a.output,dict(passed=True,cases=cases,bounds_and_invalid_slot_passed=True,resident_probe=probe.tolist(),binary_sha256=sha(binary),
        seconds=time.time()-began,raw_ssd_access=False,full_training_kernels_tested=False,
        scope='Shared exact-row lookup with mapped CPU features and existing GPU resident probe; native NVMe smoke and four-arm trajectory parity still required'))

if __name__=='__main__':main()
