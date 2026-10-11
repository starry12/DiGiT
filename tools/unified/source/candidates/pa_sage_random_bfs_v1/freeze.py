import ast
import time
import unittest
from .common import *


def main():
    require(BASE.verify()==PARENT_SHA,'Parent changed before freeze');binary_receipt()
    suite=unittest.defaultTestLoader.loadTestsFromName(MODULE+'.tests')
    result=unittest.TextTestRunner(verbosity=2).run(suite);require(result.wasSuccessful(),'CPU tests failed')
    for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
    from ae.pa_sage.common import source_config
    inputs=[ROOT/'configs/papers/protocol.json',ROOT/cfg()['base_layout']/'orders.json',
            Path(source_config()['splits']['train']['path'])]
    files={str(f.relative_to(HERE)):sha(f) for f in sorted(HERE.glob('*')) if f.is_file() and f.name!='manifest.json'}
    write(HERE/'manifest.json',dict(parent_sha256=PARENT_SHA,files=files,random_inputs={str(f):sha(f) for f in inputs},created_unix=time.time()))
    source=verify();write(OUT/'cpu_checks.json',dict(passed=True,source_sha256=source,unit_tests=result.testsRun,native_binary_reused=True,gpu_accepted=False,ssd_accepted=False))
    from .dataset import comparison_roots
    write(OUT/'root_comparison.json',comparison_roots());print(source)


if __name__=='__main__':main()
