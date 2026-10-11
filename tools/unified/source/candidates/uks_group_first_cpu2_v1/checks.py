import unittest
from .tests import Checks
from .tests_extra import Integration
from .tests_affinity import Affinity
from .common import OUT,write


def main():
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Checks,Integration,Affinity))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(OUT/'cpu_checks.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),gpu_used=False,raw_ssd_access=False,
        native_training_accepted=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
