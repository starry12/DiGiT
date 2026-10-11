"""Reuse the frozen, accepted two-arm implementation and CUDA parity."""
from candidates.ukl_common_gpu_v26.qualification import require_smoke as accepted_smoke
from candidates.ukl_cl_five_v1.freeze import verify

def prior_accepted():
    return verify()

def require_smoke():
    from candidates.large_graph_fast_v1.gpu_check import require_accepted
    require_accepted()
    from candidates.ukl_compact_cover_v1.gpu_check import require_accepted
    require_accepted()
    verify()
    from candidates.group_cache_migration_v1.gpu_check import require_accepted
    require_accepted()
    return accepted_smoke()
