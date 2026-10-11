"""Source/build/import and CPU-only regression checks; no CUDA initialization."""
import unittest
from .common import *


def main():
    setup();source=verify();check_inputs(read(OUT/'inputs.json'))
    write(OUT/'cpu_checks.json',dict(passed=False,source_sha256=source))
    for path in HERE.glob('*.py'):compile(path.read_text(),str(path),'exec')
    import torch
    import BAM_Feature_Store as native
    from .loader import PairOverlapGIDS
    from . import tests
    from candidates.pa_sage_512b_overlap_v1.window_tests import WindowTests
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==BINARY,'Wrong backend')
    require(native.PA_PAIR_ASYNC_API==native.ASYNC_FEATURE_READ_API==native.FIFO_WINDOW_512B_API==1,'Wrong native ABI')
    for method in ('read_feature_pair_async','get_pair_counts','collect_feature_read','prepare_async_reads'):
        require(hasattr(native.BAM_Feature_Store_float,method),'Missing native API: '+method)
    suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(WindowTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    require(not torch.cuda.is_initialized(),'CPU check opened CUDA')
    require(verify()==source,'Source changed during checks')
    write(OUT/'cpu_checks.json',dict(passed=result.wasSuccessful(),source_sha256=source,
        binary_sha256=sha(BINARY),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        gpu_used=False,raw_ssd_used=False,native_import_verified=True,finished_unix=time.time()))
    require(result.wasSuccessful(),'CPU regression failed')


if __name__=='__main__':main()
