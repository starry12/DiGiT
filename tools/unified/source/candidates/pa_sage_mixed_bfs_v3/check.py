"""CPU-only source, identity, order, schedule and evaluation-contract checks."""
import unittest,subprocess
from .common import *


def main():
    setup();source=verify();check_inputs(read(OUT/'inputs.json'))
    write(OUT/'cpu_checks.json',dict(passed=False,source_sha256=source))
    for path in HERE.glob('*.py'):compile(path.read_text(),str(path),'exec')
    import torch
    import BAM_Feature_Store as native
    from . import worker,gids_worker,training,loader,evaluation,tests
    from .admission import estimate
    from .graph_io import overlay_manifest
    binding=read(OUT/'inputs.json')
    plans={arm:estimate(read(HERE/'protocol.json'),overlay_manifest(binding),arm) for arm in ('gids','digit')}
    write(OUT/'admission_plans.json',plans)
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==BINARY,'Wrong mixed backend')
    require(native.PA_PAIR_ASYNC_API==1,'Missing mixed API')
    for variant in ('gids_ae','random_ae'):
        subprocess.run([PY,'-B','-m',MODULE+'.namespace','--variant',variant],cwd=ROOT,check=True)
    from ae.pa_sage.common import splits
    from ae.pa_sage.trace_cache import PreparedEvaluationTrace
    from .order import prepare_orders
    import numpy as np
    p=read(HERE/'protocol.json');train=np.sort(splits('train'));prov=read(HERE/'provenance.json')
    for mode in VARIANTS:
        for smoke in (True,False):
            orders=prepare_orders(train,mode,smoke,p)
            require(len(orders)==(2 if smoke else 20),'Wrong order extent')
    trace_budgets={}
    for split,desc in prov['traces'].items():
        trace=PreparedEvaluationTrace(desc['path'],desc['sha256'])
        trace.validate_contract(p['nodes'],p['edges'],p['fanouts'],p['batch_size'],splits(split))
        require(trace.budget['required_bytes']<=8*2**30,'Evaluation exceeds host reserve')
        trace_budgets[split]=trace.budget
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    require(not torch.cuda.is_initialized(),'CPU check initialized GPU')
    require(verify()==source,'Source changed')
    write(OUT/'cpu_checks.json',dict(passed=result.wasSuccessful(),source_sha256=source,binary_sha256=sha(BINARY),
        tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        order_checks=66,trace_budgets=trace_budgets,native_import=True,gids_native_import=True,gpu_used=False,raw_ssd_used=False))
    require(result.wasSuccessful(),'CPU tests failed')


if __name__=='__main__':main()
