"""Partition one fixed post-graph allowance; never count auxiliary arrays twice."""

def auxiliary_budget(extra, specs):
    auxiliary = sum((v['length'] + 4095) // 4096 * 4096 for v in specs.values())
    if auxiliary < 0 or auxiliary > extra:
        raise ValueError('Auxiliary arrays exceed the existing post-graph allowance')
    return dict(auxiliary_bytes=auxiliary, after_aux_bytes=extra-auxiliary,
                post_graph_allowance_bytes=extra)


def require_available(available, remaining, reserve):
    required = remaining + reserve
    if min(available, remaining, reserve) < 0:
        raise ValueError('Negative host memory budget')
    if available < required:
        raise RuntimeError('Remaining host allocation admission: available={:.3f} GiB, '
                           'remaining={:.3f} GiB, reserve={:.3f} GiB, required={:.3f} GiB'.format(
                               available/2**30, remaining/2**30, reserve/2**30, required/2**30))
