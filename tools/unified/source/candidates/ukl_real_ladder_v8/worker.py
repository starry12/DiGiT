"""Read an existing graph range into owned anonymous pages and validate GPU lifecycle."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

from . import protocol as P
from .memory import Arena, Registration, RETAINED, CHUNK


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(str(tmp), str(path))


def limits():
    b = P.budget()
    cg = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines()
              if x.startswith('0::'))
    folder = Path('/sys/fs/cgroup') / cg.lstrip('/')
    values = {k: int((folder/k).read_text()) for k in
              ('memory.max', 'memory.high', 'memory.swap.max', 'pids.max')}
    quota, period = map(int, (folder/'cpu.max').read_text().split())
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    expected = {'memory.max': b['memory_max'], 'memory.high': b['memory_high'],
                'memory.swap.max': 0, 'pids.max': 64}
    if values != expected or not 0 < quota <= period or soft != b['memlock'] or hard != soft:
        raise RuntimeError('Effective cgroup/MEMLOCK limits differ from plan')
    _, source_device = P.source_device()
    device_limits = dict((line.split()[0], dict(v.split('=') for v in line.split()[1:]))
                         for line in (folder/'io.max').read_text().splitlines())
    if not 0 < int(device_limits[source_device]['rbps']) <= P.READ_RATE:
        raise RuntimeError('Direct-read I/O limit missing or too high')
    values.update(cgroup=cg, cpu_quota=quota, cpu_period=period,
                  memlock_soft=soft, memlock_hard=hard, io_max=device_limits)
    return values


def unmapped(address, size):
    for line in Path('/proc/self/maps').read_text().splitlines():
        lo, hi = (int(v, 16) for v in line.split()[0].split('-'))
        if max(lo, address) < min(hi, address+size):
            return False
    return True


def gpu_readback(arena, reg, check, progress):
    api = reg.api
    for name, args in {'cudaMalloc': [C.POINTER(C.c_void_p), C.c_size_t],
                       'cudaMemcpy': [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
                       'cudaFree': [C.c_void_p]}.items():
        getattr(api.rt, name).argtypes = args
        getattr(api.rt, name).restype = C.c_int
    device = C.c_void_p()
    host = (C.c_char*CHUNK)()
    api.call('cudaMalloc', C.byref(device), CHUNK)
    digest = hashlib.sha256()
    try:
        for offset in range(0, arena.size, CHUNK):
            check()
            size = min(CHUNK, arena.size-offset)
            api.call('cudaMemcpy', device, reg.pointers['data']+offset, size, 3)
            api.call('cudaMemcpy', C.addressof(host), device, size, 2)
            api.synchronize()
            block = bytes(host[:size])
            if block != arena.mm[offset:offset+size]:
                raise RuntimeError('GPU readback mismatch')
            digest.update(block)
            progress(offset+size)
    finally:
        api.synchronize()
        api.call('cudaFree', device)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier-gib', type=int, choices=(2, 4, 8), required=True)
    P.configure_tier(parser.parse_args().tier_gib)
    if os.environ.get('UKL_V8_BOUNDED_WORKER') != '1':
        raise RuntimeError('Use the bounded controller')
    out = Path(os.environ['UKL_V8_OUTPUT'])
    P.verify_manifest()
    source = P.binding()
    actual_limits = limits()
    write(out/'effective_limits.json', actual_limits)
    events = []

    def event(stage, **kwargs):
        events.append(dict(time=time.time(), stage=stage, **kwargs))
        write(out/'lifecycle.json', events)

    terminate_requested = [False]

    def check(*_):
        if terminate_requested[0] or (out/'STOP').exists():
            raise RuntimeError('Controller requested a cooperative stop')

    def stopped(signum, frame):
        # Never interrupt the registration/ownership state transition.
        terminate_requested[0] = True

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    started = time.monotonic()
    last_progress = [0]

    def progress(name, loaded, length, total_loaded, total):
        check()
        wait = total_loaded/P.READ_RATE-(time.monotonic()-started)
        if wait > 0:
            time.sleep(min(wait, 1))
        if total_loaded-last_progress[0] >= 64*P.MIB or total_loaded == total:
            event('load_progress', loaded_bytes=total_loaded, remaining_bytes=total-total_loaded)
            last_progress[0] = total_loaded

    arena = reg = None
    try:
        event('load', extent_bytes=P.EXTENT, buffered_fallback=False)
        arena = Arena({'data': source})
        arena.load(progress)
        check()
        event('loaded', digests=arena.digests, direct_io=True)
        address, size = arena.address, arena.size
        event('register')
        reg = Registration(arena)
        event('registered')
        actual = gpu_readback(arena, reg, check, lambda n: None)
        expected = arena.digests['data']
        if actual != expected:
            raise RuntimeError('Full range digest mismatch')
        event('unregister')
        reg.close(); reg = None
        event('unregistered')
        arena.close(); arena = None
        if RETAINED or not unmapped(address, size):
            raise RuntimeError('Anonymous allocation not released')
        event('complete', vma_absent=True)
        write(out/'worker.json', dict(passed=True, tier_gib=P.TIER, extent_bytes=size,
            loaded_bytes=size, gpu_readback_bytes=size, source_sha256=expected,
            gpu_sha256=actual, direct_io=True, normal_unregister=True,
            anonymous_memory_released=True, vma_absent_after_release=True,
            limits_verified_before_cuda=True, effective_limits=actual_limits,
            maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            source_identity=source['identity'], full_graph_enabled=False,
            scope='Existing UKL adjacency byte range; no sampler/model/SSD validation'))
    finally:
        if reg is not None:
            reg.close()
        if arena is not None:
            arena.close()


if __name__ == '__main__':
    main()
