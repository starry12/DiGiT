"""Validate eight real UKL arrays in owned anonymous memory, without training."""
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
    """Compare every allocated byte, hashing only each array's logical payload."""
    api = reg.api
    for name, args in {'cudaMalloc': [C.POINTER(C.c_void_p), C.c_size_t],
                       'cudaMemcpy': [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
                       'cudaFree': [C.c_void_p]}.items():
        getattr(api.rt, name).argtypes = args
        getattr(api.rt, name).restype = C.c_int
    device = C.c_void_p()
    host = (C.c_char * CHUNK)()
    check()
    api.call('cudaMalloc', C.byref(device), CHUNK)
    arrays, verified = {}, 0
    try:
        for name, spec in arena.specs.items():
            logical, allocated = arena.lengths[name], arena.allocated_lengths[name]
            if set(reg.pointers) != set(arena.specs):
                raise RuntimeError('CUDA array pointer names differ from loaded arrays')
            digest = hashlib.sha256()
            for offset in range(0, allocated, CHUNK):
                check()
                size = min(CHUNK, allocated - offset)
                payload = max(0, min(size, logical - offset))
                start = arena.offsets[name] + offset
                cpu = arena.mm[start:start + size]
                if payload < size and any(cpu[payload:]):
                    raise RuntimeError('CPU tail padding is not zero: ' + name)
                api.call('cudaMemcpy', device, reg.pointers[name] + offset, size, 3)
                api.call('cudaMemcpy', C.addressof(host), device, size, 2)
                api.synchronize()
                block = C.string_at(C.addressof(host), size)
                if payload < size and any(block[payload:]):
                    raise RuntimeError('GPU tail padding is not zero: ' + name)
                if block != cpu:
                    raise RuntimeError('GPU readback mismatch: ' + name)
                digest.update(block[:payload])
                verified += size
                progress(name, offset + size, allocated, verified, arena.size)
            actual, expected = digest.hexdigest(), spec.get('sha256')
            if actual != arena.digests[name]:
                raise RuntimeError('Logical array digest mismatch: ' + name)
            if expected is not None and actual != expected:
                raise RuntimeError('Frozen whole-file digest mismatch: ' + name)
            arrays[name] = dict(path=str(spec['path']), offset=spec['offset'],
                logical_length=logical, allocated_bytes=allocated,
                arena_offset=arena.offsets[name], padding_bytes=allocated-logical,
                cpu_sha256=arena.digests[name], gpu_sha256=actual,
                cpu_padding_zero=True, gpu_padding_zero=True,
                source_identity=list(arena.source_identities[name]),
                expected_sha256=expected,
                expected_sha256_matched=True if expected is not None else None)
        if verified != arena.size:
            raise RuntimeError('Incomplete allocated arena GPU readback')
    finally:
        api.synchronize()
        api.call('cudaFree', device)
    return arrays


def progress_callback(event, check, phase, rate=None):
    """Bound loading rate and emit sparse progress plus every array boundary."""
    started = time.monotonic()
    last = [0]

    def progress(name, done, length, total_done, total):
        check()
        if rate is not None:
            while True:
                wait = total_done / rate - (time.monotonic() - started)
                if wait <= 0:
                    break
                time.sleep(min(wait, 0.2))
                check()
        if total_done - last[0] >= 256 * P.MIB or done == length:
            event(phase + '_progress', array=name, array_bytes_done=done,
                  array_bytes=length, allocated_bytes_done=total_done,
                  allocated_bytes=total, remaining_bytes=total-total_done)
            last[0] = total_done
    return progress


def _revalidate(sources):
    if P.binding() != sources:
        raise RuntimeError('Graph source binding changed during worker lifecycle')


def execute(out):
    P.verify_manifest()
    sources = P.binding()
    if len(sources) != 8:
        raise RuntimeError('Exactly eight named graph arrays are required')
    hashes = sum(spec.get('sha256') is not None for spec in sources.values())
    if hashes != (8 if P.STAGE == 'full' else 7):
        raise RuntimeError('Frozen whole-file SHA256 coverage differs from selected stage')
    actual_limits = limits()
    write(out/'effective_limits.json', actual_limits)
    events = []

    def event(stage, **kwargs):
        events.append(dict(time=time.time(), stage=stage, selected_stage=P.STAGE, **kwargs))
        write(out/'lifecycle.json', events)

    terminate_requested = [False]
    arena = reg = None

    def check():
        if terminate_requested[0] or (out/'STOP').exists():
            raise RuntimeError('Controller requested a cooperative stop')
        if arena is not None:
            arena.check_admission()

    def stopped(signum, frame):
        # Do not interrupt a CUDA registration/ownership state transition.
        terminate_requested[0] = True

    previous_term = signal.signal(signal.SIGTERM, stopped)
    previous_int = signal.signal(signal.SIGINT, stopped)
    try:
        check()
        event('load', arena_bytes=P.EXTENT, arrays=list(sources), buffered_fallback=False)
        arena = Arena(sources, max_bytes=P.EXTENT)
        if arena.size != P.EXTENT:
            raise RuntimeError('Allocated arena differs from fixed stage plan')
        arena.load(progress_callback(event, check, 'load', P.READ_RATE))
        check()
        logical = sum(arena.lengths.values())
        if arena.loaded_bytes != arena.size or arena.loaded_payload_bytes != logical:
            raise RuntimeError('Incomplete physical/logical array loading')
        _revalidate(sources)
        event('loaded', digests=arena.digests, loaded_bytes=arena.loaded_bytes,
              loaded_payload_bytes=logical, direct_io=True, source_revalidated=True)
        address, size = arena.address, arena.size
        event('register')
        reg = Registration(arena)
        event('registered')
        arrays = gpu_readback(arena, reg, check, progress_callback(event, check, 'gpu_readback'))
        check()
        event('unregister')
        reg.close(); reg = None
        event('unregistered')
        arena.close(); arena = None
        if RETAINED or not unmapped(address, size):
            raise RuntimeError('Anonymous allocation not released')
        _revalidate(sources)
        event('complete', vma_absent=True, source_revalidated=True)
        write(out/'worker.json', dict(passed=True, stage=P.STAGE, arena_bytes=size,
            logical_bytes=logical, loaded_bytes=size, loaded_payload_bytes=logical,
            gpu_readback_bytes=logical, gpu_readback_allocated_bytes=size,
            padding_bytes=size-logical, arrays=arrays, direct_io=True,
            normal_unregister=True, anonymous_memory_released=True,
            vma_absent_after_release=True, limits_verified_before_cuda=True,
            effective_limits=actual_limits, source_revalidated_before_cuda=True,
            source_revalidated_after_release=True,
            maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            full_graph_load=P.STAGE == 'full', full_graph_enabled=False,
            graph_sampling_enabled=False, model_enabled=False, raw_ssd_access=False,
            scope='Eight real UKL graph arrays; CUDA readback and lifecycle only'))
    finally:
        try:
            if reg is not None:
                reg.close()
            if arena is not None:
                arena.close()
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('64', 'full'), required=True)
    P.configure_stage(parser.parse_args().stage)
    if os.environ.get('UKL_V9_BOUNDED_WORKER') != '1':
        raise RuntimeError('Use the bounded controller')
    execute(Path(os.environ['UKL_V9_OUTPUT']))


if __name__ == '__main__':
    main()
