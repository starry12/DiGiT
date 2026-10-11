"""Reuse the frozen, accepted two-arm implementation and CUDA parity."""
from candidates.ukl_common_gpu_v26.qualification import require_smoke as accepted_smoke
from candidates.ukl_cl_five_v1.freeze import verify

def prior_accepted():
    return verify()

def require_smoke():
    verify()
    return accepted_smoke()
