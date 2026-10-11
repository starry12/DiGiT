"""Partition one fixed post-graph allowance; never count auxiliary arrays twice."""
import time
from pathlib import Path


def available_memory():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


class HostAdmission:
    def __init__(self, check, reserve, *, read=available_memory, clock=time.monotonic):
        self.check, self.reserve, self.read, self.clock = check, reserve, read, clock
        self.last_time = None
        self.last_remaining = None

    def __call__(self, remaining=0, *, force=False):
        self.check()
        if type(remaining) is not int or remaining < 0:
            raise ValueError('Invalid remaining host allocation')
        now = self.clock()
        if (force or self.last_time is None or now-self.last_time >= 1.0
                or remaining > self.last_remaining):
            require_available(self.read(), remaining, self.reserve)
            self.last_time = now
        self.last_remaining = remaining

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
