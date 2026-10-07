"""Run CPU model regressions using the parent model source shipped in this repo."""
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ae
# The clean artifact retains large-dataset dependencies in runtime_sources.
# Extend only this test process's package path; installed workers use their
# immutable parent snapshots and need no import changes.
ae.__path__.append(str(ROOT / 'tools/ig_performance/runtime_sources/ae'))

if __name__ == '__main__':
    module = importlib.import_module('tools.performance_models.deployment.runtime.tests')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    raise SystemExit(0 if result.wasSuccessful() else 1)
