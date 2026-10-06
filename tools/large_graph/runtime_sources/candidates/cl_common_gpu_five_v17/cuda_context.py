"""Materialize a current CUDA context before verifying its full device UUID."""
from candidates.ukl_training_native_v15r11.gpu_identity import verify_cuda_uuid

def initialize(expected):
    import torch
    torch.cuda.set_device(0)
    torch.cuda.init()
    # Torch lazy init alone may leave cuCtxGetCurrent() null. A one-byte tensor
    # forces the same allocator/context path later used by actual workloads.
    scratch=torch.empty(1,dtype=torch.uint8,device='cuda:0')
    try:
        torch.cuda.synchronize()
        return verify_cuda_uuid(expected)
    finally:
        del scratch
