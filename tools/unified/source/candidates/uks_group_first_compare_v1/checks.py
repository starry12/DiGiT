import unittest
from .tests import Checks
from .tests_extra import Integration
from .common import OUT,write

def main():
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Checks,Integration))
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    write(OUT/'cpu_checks.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,
         raw_ssd_access=False,native_training_accepted=False))
    raise SystemExit(0 if r.wasSuccessful() else 1)

if __name__=='__main__':main()
