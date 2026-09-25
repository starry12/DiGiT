"""Import the provisioned runtime without creating a CUDA context or opening SSD."""
import importlib,json,os,sys
from pathlib import Path
root=Path('/home/embed/digit');sys.path.insert(0,str(root))
os.environ['DIGIT_LAYOUT_PROTOCOL']=str(root/'results/pa_sage_layout_shared_resume_20260925_v3/protocols/g2_r20.json')
from candidates.pa_sage_layout_shared_resume_v3.common import setup,verify,cfg
setup();identity=verify();cfg()
import torch,dgl,runner,digit_paths
from digit import artifacts,io_geometry,DiGiTSamplerCUDA
import BAM_Feature_Store
modules=[runner,digit_paths,artifacts,io_geometry,DiGiTSamplerCUDA,sys.modules['BAM_Feature_Store.BAM_Feature_Store']]
for name in ('worker','training_loop','validation','admission','binding','aggregate'):
    modules.append(importlib.import_module('candidates.pa_sage_layout_shared_resume_v3.'+name))
for module in modules:Path(module.__file__).resolve().relative_to(root)
assert digit_paths.project_root()==root
assert artifacts._digit_root==root and io_geometry._digit_root==root
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,candidate_sha256=identity,modules=[m.__name__ for m in modules],
    cuda_initialized=False,native_acceptance=False)))
