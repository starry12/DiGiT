"""Fresh-process CPU-only import checks for the two different native backends."""
import argparse
from .common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=VARIANTS,required=True)
    variant=parser.parse_args().variant
    if variant=='gids_decay':gids_reference.setup()
    else:setup()
    import torch
    import BAM_Feature_Store as native
    from .loader import GIDSEpochLoader,MultiEpochGIDS
    actual=Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()
    require(actual==binary_for(variant),'Wrong native module')
    require(native.FIFO_WINDOW_512B_API==1,'Wrong baseline/window ABI')
    if variant=='gids_decay':
        require(not hasattr(native,'PA_PAIR_ASYNC_API'),'GIDS accidentally uses the mixed backend')
    else:require(native.PA_PAIR_ASYNC_API==1,'DiGiT does not use mixed backend')
    require(not torch.cuda.is_initialized(),'Namespace check initialized CUDA')
    print(json.dumps(dict(passed=True,variant=variant,binary=str(actual),gpu_used=False)))


if __name__=='__main__':main()
