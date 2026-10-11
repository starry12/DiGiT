"""CL safe-load plan and bounded CPU checks; no full/GPU/service command."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import unittest
from . import protocol as P


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','check-cpu'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve old evidence')
    if args.action=='plan':
        value=dict(binding=P.csc_binding(),budget=P.budget(),raw_ssd_access=False,gpu_called=False)
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide GPUs for CPU checks')
        if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
            raise RuntimeError('Set CPU thread limits to one')
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))});os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
        start=time.monotonic()
        from candidates.ukl_runtime_prepare_v11r1 import protocol as U
        before=U.verify_manifest()
        from . import test_loader as T
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(T))
        unchanged=U.verify_manifest()==before
        value=dict(passed=r.wasSuccessful() and unchanged,tests=r.testsRun,failures=len(r.failures),
                    errors=len(r.errors),seconds=time.monotonic()-start,
                    maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    protected_manifest_sha256=before,protected_manifest_unchanged=unchanged,
                    gpu_called=False,raw_ssd_access=False,full_cl_loaded=False)
    with args.output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('binding','budget')}))
    if args.action=='check-cpu':raise SystemExit(not value['passed'])


if __name__=='__main__':main()
