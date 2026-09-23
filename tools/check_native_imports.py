"""Import real PA worker dependencies with CUDA hidden, without dataset or SSD access."""
import importlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DGLBACKEND'] = 'pytorch'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from training.sage.common import setup
    setup()
    modules = {}
    names = ('BAM_Feature_Store', 'digit.DiGiTSamplerCUDA', 'digit.DiGiTOutputCUDA',
             'GIDS', 'GIDS.breakdown', 'runner',
             'training.sage.worker', 'training.gcn.worker', 'training.gat.worker')
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if ROOT not in path.parents:
            raise RuntimeError('Dependency escaped this copy: ' + str(path))
        modules[name] = str(path.relative_to(ROOT))
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Import check unexpectedly initialized CUDA')
    print(json.dumps(dict(passed=True, modules=modules, gpu_workload=False,
                         raw_ssd_access=False, native_acceptance=False), indent=2))

if __name__ == '__main__':
    main()
