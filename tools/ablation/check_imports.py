"""Check runtime roots in a fresh child; no CUDA context or device access."""
import importlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from candidates.pa_sage_ablation_graph_v4.common import setup
setup()
import torch, dgl, runner, digit_paths
from digit import artifacts, gpu_admission, io_geometry, ssd_payload, DiGiTSamplerCUDA
import BAM_Feature_Store
from ae import common
assert digit_paths.project_root() == root and common.ROOT == root
for module in (artifacts, gpu_admission, io_geometry):
    assert module._digit_root == root, (module.__name__, module._digit_root)
modules = [runner, digit_paths, artifacts, gpu_admission, io_geometry, ssd_payload, DiGiTSamplerCUDA,
           sys.modules['BAM_Feature_Store.BAM_Feature_Store']]
for version in ('cache_v3', 'graph_v4'):
    for name in ('worker', 'training_loop', 'cache', 'validation', 'admission'):
        modules.append(importlib.import_module('candidates.pa_sage_ablation_' + version + '.' + name))
for module in modules: Path(module.__file__).resolve().relative_to(root)
assert not torch.cuda.is_initialized(), 'CPU import check initialized CUDA'
print(json.dumps(dict(passed=True, root=str(root), modules=[m.__name__ for m in modules],
    cuda_initialized=False, training_started=False)))
