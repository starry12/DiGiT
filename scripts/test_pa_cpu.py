"""PA CPU regressions using one immutable graph-size metadata fixture.

Only graph-size and preparation-receipt metadata reads are substituted. No graph/payload is attached,
no CUDA computation is launched, and the estimate is not native acceptance.
"""
import json,sys,unittest
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
(ROOT/'results').mkdir(exist_ok=True)
from training.sage.common import setup,cfg
setup()
import torch
torch.set_num_threads(1)
from training.sage import admission
from training.gcn import tests as gcn
from training.gat import tests as gat
from training.gcn import inputs as gcn_inputs
from training.gat import inputs as gat_inputs
from contextlib import ExitStack
original=admission.read
expected=ROOT/cfg()['base_layout']/'final/bundle/manifest.json'
prepared=ROOT/cfg()['data']/'prepared.json'
original_sha=gcn_inputs.sha
def fixture_read(path):
    if Path(path)==expected:
        return json.loads((ROOT/'tests/fixtures/pa_budget_manifest.json').read_text())
    if Path(path)==prepared:
        return json.loads((ROOT/'tests/fixtures/pa_prepared.json').read_text())
    return original(path)
def fixture_sha(path):
    return original_sha(ROOT/'tests/fixtures/pa_budget_manifest.json') if Path(path)==expected else original_sha(path)
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (gcn,gat)])
with ExitStack() as stack:
    for module in (admission,gcn_inputs,gat_inputs):
        stack.enter_context(mock.patch.object(module,'read',side_effect=fixture_read))
    for module in (gcn_inputs,gat_inputs):
        stack.enter_context(mock.patch.object(module,'sha',side_effect=fixture_sha))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),budget_metadata_fixture=True,raw_ssd_access=False,cuda_compute=False),indent=2))
sys.exit(not result.wasSuccessful())
