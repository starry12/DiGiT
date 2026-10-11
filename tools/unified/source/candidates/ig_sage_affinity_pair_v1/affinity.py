"""Original GIDS scheduling; original DiGiT worker threads on logical CPU2."""
import os
from candidates.ig_sage_host_opt_v1.affinity import snapshot


def apply_affinity(arm):
    if arm not in ('gids', 'digit_full'):
        raise ValueError('Unknown arm')
    before = snapshot()
    bound = arm == 'digit_full'
    if bound:
        if 2 not in os.sched_getaffinity(0):
            raise RuntimeError('CPU2 outside worker CPU set')
        for tid in before['thread_masks']:
            try:
                os.sched_setaffinity(int(tid), {2})
            except ProcessLookupError:
                pass
    value = dict(arm=arm, policy='one_core' if bound else 'unchanged',
                 cpu=2 if bound else None, before=before, after=snapshot())
    validate_affinity(dict(initial=value, final=value['after']), arm)
    return value


def validate_affinity(value, arm):
    from .common import require
    initial = value['initial']
    bound = arm == 'digit_full'
    require(initial['arm'] == arm and initial['policy'] == ('one_core' if bound else 'unchanged')
            and initial['cpu'] == (2 if bound else None), 'Wrong CPU affinity policy')
    before = initial['before']['thread_masks']
    require(bool(before), 'Missing original CPU masks')
    original_masks = list(before.values())
    for snapshot_ in (initial['after'], value['final']):
        masks = snapshot_['thread_masks']
        require(bool(masks), 'Missing worker CPU affinity evidence')
        if bound:
            require(all(mask == [2] for mask in masks.values()), 'Worker threads escaped CPU2')
        else:
            require(all(mask == before[tid] if tid in before else mask in original_masks
                        for tid, mask in masks.items()), 'GIDS default CPU mask changed')
    return True
