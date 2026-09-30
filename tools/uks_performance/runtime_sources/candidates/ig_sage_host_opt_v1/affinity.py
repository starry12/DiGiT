"""Only modify this worker's existing/future thread affinities; no governor changes."""
import os
from pathlib import Path


def snapshot():
    masks={}
    for task in Path('/proc/self/task').iterdir():
        try:masks[task.name]=sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:pass
    path=Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq')
    return dict(thread_masks=masks,cpu2_frequency_khz=int(path.read_text()) if path.exists() else None)


def apply_affinity(variant):
    if variant not in ('legacy','affinity','compact','combined'):
        raise ValueError('Unknown variant')
    before=snapshot();bound=variant in ('affinity','combined')
    if bound:
        if 2 not in os.sched_getaffinity(0):raise RuntimeError('CPU2 outside worker CPU set')
        for tid in before['thread_masks']:
            try:os.sched_setaffinity(int(tid),{2})
            except ProcessLookupError:pass
    after=snapshot()
    if bound and any(mask!=[2] for mask in after['thread_masks'].values()):
        raise RuntimeError('Incomplete worker thread affinity')
    return dict(variant=variant,policy='one_core' if bound else 'unchanged',cpu=2 if bound else None,before=before,after=after)
