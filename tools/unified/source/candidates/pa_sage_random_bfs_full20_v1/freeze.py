import ast
import time
import unittest
from .common import *

def main():
    require(BASE.verify()==PARENT_SHA,'Parent changed');binary_receipt()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(MODULE+'.tests'))
    require(result.wasSuccessful(),'CPU tests failed')
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    dependencies=['ae/pa_sage/trace_cache.py','ae/pa_sage/common.py','ae/igb/models.py','ae/common.py',
                  'candidates/pa_sage_bidir_native_v2/runtime/digit/eval_trace.py',
                  'candidates/pa_sage_cache_policy_v1/training.py']
    inputs={str(ROOT/v['path']/'manifest.json'):v['sha256'] for v in read(HERE/'evaluation_inputs.json').values()}
    files={str(f.relative_to(HERE)):sha(f) for f in sorted(HERE.iterdir()) if f.is_file() and f.name!='manifest.json'}
    write(HERE/'manifest.json',dict(parent_sha256=PARENT_SHA,files=files,inputs=inputs,
        dependencies={f:sha(ROOT/f) for f in dependencies},created_unix=time.time()))
    source=verify();write(OUT/'cpu_checks.json',dict(passed=True,source_sha256=source,unit_tests=result.testsRun,
        native_binary_reused=True,all_twenty_orders_checked=True,gpu_accepted=False,full20_accepted=False))
    print(source)

if __name__=='__main__':main()
