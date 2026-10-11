"""Run bounded CPU fixtures, never launch preparation or production training."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import unittest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing CPU evidence: '+str(args.output))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES empty for CPU-only checks')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise RuntimeError('Set '+name+'=1')
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    os.nice(19)
    start=time.monotonic()
    from . import test_training as T
    from candidates.ukl_runtime_prepare_v11r1 import protocol as P
    protected=P.verify_manifest()
    suite=unittest.defaultTestLoader.loadTestsFromModule(T)
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged=P.verify_manifest()==protected
    value=dict(passed=r.wasSuccessful() and unchanged,tests=r.testsRun,
               failures=len(r.failures),errors=len(r.errors),seconds=time.monotonic()-start,
               maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               cuda_initialized=T.torch.cuda.is_initialized(),raw_ssd_access=False,
               protected_manifest_sha256=protected,protected_manifest_unchanged=unchanged,
               reports=T.REPORTS)
    with args.output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='reports'}))
    raise SystemExit(not value['passed'])


if __name__=='__main__':main()
