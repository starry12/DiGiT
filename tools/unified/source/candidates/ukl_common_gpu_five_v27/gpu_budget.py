"""Check remaining allocations after framework initialization, without CUDA I/O."""

def admit_initialized(budget, free_bytes, total_bytes):
    components = budget['gpu_components']
    initialized_allowance = components['framework_reserve']
    remaining = sum(v for k, v in components.items() if k != 'framework_reserve')
    margin = budget['gpu_free_min'] - budget['gpu_accounted']
    if (sum(components.values()) != budget['gpu_accounted'] or margin < 0
            or initialized_allowance <= 0 or not 0 <= free_bytes <= total_bytes):
        raise RuntimeError('Invalid initialized GPU budget')
    required = remaining + margin
    receipt = dict(passed=free_bytes >= required, phase='after_model_initialization',
                   free_bytes=free_bytes, total_bytes=total_bytes,
                   required_free_bytes=required, remaining_accounted_bytes=remaining,
                   headroom_bytes=margin, framework_allowance_bytes=initialized_allowance,
                   preinitialization_free_min_bytes=budget['gpu_free_min'])
    if not receipt['passed']:
        raise RuntimeError('GPU remaining-memory admission: free={:.3f} GiB, required={:.3f} GiB '
                           '(after framework initialization)'.format(free_bytes / 2**30, required / 2**30))
    return receipt
