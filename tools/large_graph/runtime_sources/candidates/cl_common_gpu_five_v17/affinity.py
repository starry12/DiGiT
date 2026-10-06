"""The DiGiT service binds all threads to CPU2 before Python starts."""
import os
from pathlib import Path


def verify(arm):
    if arm not in ('gids','digit'):
        raise ValueError('Unknown affinity arm')
    current = sorted(os.sched_getaffinity(0))
    tasks = []
    if arm == 'digit':
        for p in Path('/proc/self/task').iterdir():
            try:
                mask = sorted(os.sched_getaffinity(int(p.name)))
            except ProcessLookupError:
                continue
            if mask != [2]:
                raise RuntimeError('DiGiT thread is not bound to CPU2')
            tasks.append(int(p.name))
        if current != [2] or not tasks:
            raise RuntimeError('DiGiT CPU2 affinity missing')
    return dict(policy='cpu2' if arm=='digit' else 'default', cpus=current,
                checked_threads=len(tasks))
