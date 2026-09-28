"""CPU-only immutable runtime import and prepared-input checks."""
import sys,tempfile,importlib,json
from pathlib import Path
sys.path.insert(0,'/home/embed/digit')
from candidates.ig_sage_pair_max5_v1.common import verify,setup,PARENT_SHA
identity=verify();setup()
for name in ('torch','dgl','IGPerfNative','sampler_config','bounded_io','uva_sampler','candidates.ig_sage_host_telemetry_v1.worker','candidates.ig_sage_pair_max5_v1.controller'):
    importlib.import_module(name)
from candidates.ig_sage_host_telemetry_v1 import inputs
with tempfile.TemporaryDirectory() as d:inputs.bind(Path(d),PARENT_SHA)
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,candidate_sha256=identity,cuda_initialized=False,native_acceptance=False)))
