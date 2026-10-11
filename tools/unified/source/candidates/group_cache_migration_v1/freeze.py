"""Seal tested migration sources and unchanged dependency receipts."""
import ast
import unittest
from .common import *
def main():
    require(not (HERE/'manifest.json').exists(),'Preserve frozen migration candidate')
    from .tests import Migration
    from .ig.monitor_tests import MonitorRevision
    from .compatibility_tests import PreparationCompatibility
    from .ig.adjacency_tests import RawAdjacencyTests
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (Migration,MonitorRevision,PreparationCompatibility,RawAdjacencyTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(),'CPU regression failed')
    from .readiness import check
    namespaces=check()
    from candidates.cache_epoch_v1.common import verify as parent_verify
    parent=parent_verify();files=dict(read(ROOT/'candidates/cache_epoch_v1/manifest.json')['files'])
    files['candidates/cache_epoch_v1/manifest.json']=parent
    for p in (ROOT/'candidates/cache_epoch_v1').rglob('*.py'):files[str(p.relative_to(ROOT))]=sha(p)
    for p in HERE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            files[str(p.relative_to(ROOT))]=sha(p)
    for p in (OUT/'build').iterdir():
        if p.suffix!='.log':files[str(p.relative_to(ROOT))]=sha(p)
    for p in (ROOT/'ae/native/igb/sampler').iterdir():
        if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
    for name in ('gpu_monitor.py','review.py','review_tests.py'):
        p=ROOT/'candidates/ig_monitor_v1'/name;files[str(p.relative_to(ROOT))]=sha(p)
    for name in ('bounded_io.py','uva_sampler.py'):
        p=ROOT/'ae/igb'/name;files[str(p.relative_to(ROOT))]=sha(p)
    write(HERE/'manifest.json',dict(files=files,parent_cache_epoch_sha256=parent,
          scope='Independent IG/UKL/CL v2 reuse, strict group priority, complete profiling epoch, one 20+300 pair'))
    digest=verify();write(OUT/'cpu_checks.json',dict(passed=True,source_sha256=digest,tests=result.testsRun,namespaces=namespaces,gpu_initialized=False,raw_ssd_access=False,native_training_accepted=False))
    print('Frozen '+digest,flush=True)
if __name__=='__main__':main()
