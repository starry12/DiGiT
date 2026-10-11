"""Bounded CPU-only cgroup-format probe: one 4 KiB direct graph read, no CUDA."""
import json
import mmap
import os
from pathlib import Path
import time

from . import protocol as P


def main():
    rows = [x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines()
            if x.startswith('0::')]
    if len(rows) != 1 or '/digit-ukl-iostat-probe-' not in rows[0]:
        raise RuntimeError('Run only inside the bounded probe service')
    cg = Path('/sys/fs/cgroup') / rows[0].lstrip('/')
    limits = {name: (cg / name).read_text() for name in
              ('memory.max', 'memory.swap.max', 'cpu.max', 'io.max')}
    if int(limits['memory.max']) > 64 * 1024**2 or int(limits['memory.swap.max']) != 0:
        raise RuntimeError('Probe memory limits missing')
    quota, period = map(int, limits['cpu.max'].split())
    if quota > period:
        raise RuntimeError('Probe CPU limit missing')
    _, device = P.source_device()
    limit_rows = {x.split()[0]: dict(t.split('=', 1) for t in x.split()[1:])
                  for x in limits['io.max'].splitlines()}
    if limit_rows.get(device, {}).get('rbps') != str(P.READ_RATE):
        raise RuntimeError('Probe source-device limit missing')
    P.binding()
    samples = []

    def observe(stage):
        samples.append(dict(stage=stage, time=time.time(), raw=(cg / 'io.stat').read_text()))

    observe('before_read')
    arena = mmap.mmap(-1, 4096, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                      prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        fd = os.open(str(P.SOURCE), os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
        try:
            count = os.preadv(fd, [arena], 0)
        finally:
            os.close(fd)
    finally:
        arena.close()
    if count != 4096:
        raise RuntimeError('Incomplete 4 KiB CPU direct read')
    observe('after_read')
    for _ in range(3):
        time.sleep(0.2)
        observe('after_counter_update')
    print(json.dumps(dict(scope='cpu_cgroup_io_format_probe', read_bytes=count,
                         direct_io=True, gpu_called=False, raw_ssd_access=False,
                         source_device=device, cgroup=rows[0], limits=limits,
                         samples=samples)), flush=True)


if __name__ == '__main__':
    main()
