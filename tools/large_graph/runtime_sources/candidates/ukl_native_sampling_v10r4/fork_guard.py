"""Prevent child creation while owned anonymous GPU input remains live.

The Python audit guard fails closed on known process creation/exec paths.
MADV_DONTFORK is an independent kernel-level inheritance boundary for each
arena. Neither mechanism is a sandbox against arbitrary native code.
"""
import ctypes as C
import os
import re
import sys
import threading


MADV_DONTFORK = 10  # Linux uapi asm-generic/mman-common.h
_HEADER = re.compile(r'^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+(\S+)\s+(\d+)(?:\s+(.*))?$')
_BLOCKED = frozenset(('os.fork', 'os.forkpty', 'os.posix_spawn',
                      'subprocess.Popen', 'os.system', 'os.exec', 'os.execve'))
_PROBE = 'digit.ukl.native_sampling.ownership_audit_probe'
_ACTIVE = {}  # Strong references survive exceptions, GC, and failed cleanup.
_LOCK = threading.RLock()
_HOOK_INSTALLED = False
_HOOK_SEEN = None
_LIBC = C.CDLL(None, use_errno=True)
_LIBC.madvise.argtypes = (C.c_void_p, C.c_size_t, C.c_int)
_LIBC.madvise.restype = C.c_int


def _audit(event, args):
    global _HOOK_SEEN
    if event == _PROBE:
        _HOOK_SEEN = args[0]
    if event in _BLOCKED and _ACTIVE:
        with _LOCK:
            guards = tuple(_ACTIVE.values())
            for guard in guards:
                guard._blocked_events += 1
        raise RuntimeError('Child creation/exec forbidden while UKL arena ownership is active: ' + event)


def _install_hook():
    global _HOOK_INSTALLED
    with _LOCK:
        if not _HOOK_INSTALLED:
            sys.addaudithook(_audit)
            nonce = object()
            sys.audit(_PROBE, nonce)
            # Another hook may silently reject addaudithook. Prove ours ran.
            if _HOOK_SEEN is not nonce:
                raise RuntimeError('Ownership audit hook installation was not verified')
            _HOOK_INSTALLED = True


def _bounds(arena, *, unloaded=False):
    from candidates.ukl_real_multi_v9.memory import Arena
    if (type(arena) is not Arena or arena.closed or arena.mm is None
            or type(arena.address) is not int or type(arena.size) is not int
            or arena.address <= 0 or arena.size <= 0
            or arena.address % 4096 or arena.size % 4096):
        raise RuntimeError('A live exact owned v9 anonymous Arena is required')
    if unloaded and (arena.loaded or arena.loaded_bytes or arena.registered):
        raise RuntimeError('MADV_DONTFORK must be installed before loading/registration')
    return arena.address, arena.size


def _overlaps(address, size, path='/proc/self/maps'):
    end = address + size
    with open(path, 'r') as stream:
        for line in stream:
            match = _HEADER.match(line.rstrip('\n'))
            if match is not None:
                start, stop = int(match[1], 16), int(match[2], 16)
                if start < end and stop > address:
                    return True
    return False


def _vmas(address, size, path='/proc/self/smaps'):
    """Stream only metadata; require contiguous private anonymous dc coverage."""
    end, cursor, records, current = address + size, address, [], None
    with open(path, 'r') as stream:
        for line in stream:
            match = _HEADER.match(line.rstrip('\n'))
            if match is not None:
                if current is not None:
                    raise RuntimeError('Relevant VMA lacks VmFlags')
                start, stop = int(match[1], 16), int(match[2], 16)
                if start >= end:
                    break
                if stop <= address:
                    continue
                if (start > cursor or stop <= cursor or match[3] != 'rw-p'
                        or match[4] != '00:00' or match[5] != '0'
                        or match[6] not in (None, '')):
                    raise RuntimeError('Arena range is not contiguous private anonymous memory')
                current = dict(start=start, end=stop)
            elif current is not None and line.startswith('VmFlags:'):
                flags = line.split()[1:]
                if 'dc' not in flags:
                    raise RuntimeError('Arena VMA lacks MADV_DONTFORK (VmFlags dc)')
                if {'io', 'pf', 'sh'} & set(flags):
                    raise RuntimeError('Arena VMA has I/O, PFN, or shared flags')
                current['vmflags'] = flags
                records.append(current)
                cursor = min(end, current['end'])
                current = None
    if current is not None or cursor != end or not records:
        raise RuntimeError('Could not verify every arena byte as MADV_DONTFORK')
    return dict(address=address, bytes=size, verified_bytes=size,
                madv_dontfork=True, vmflags_dc=True, anonymous_private=True,
                vmas=records)


def dontfork_arena(arena):
    """Set DONTFORK before any load/register, then independently read smaps."""
    address, size = _bounds(arena, unloaded=True)
    if _LIBC.madvise(address, size, MADV_DONTFORK) != 0:
        code = C.get_errno()
        raise OSError(code, os.strerror(code))
    return assert_dontfork(arena)


def assert_dontfork(arena):
    address, size = _bounds(arena)
    return _vmas(address, size)


class OwnershipGuard:
    """Explicit release only: failed cleanup must never disable the guard."""

    def __init__(self):
        self.active = False
        self.entered_pid = None
        self._tracked = {}
        self._blocked_events = 0
        self._released = False

    def enter(self):
        if self.active or self.entered_pid is not None:
            raise RuntimeError('Ownership guard may be entered exactly once')
        _install_hook()
        with _LOCK:
            self.entered_pid = os.getpid()
            self.active = True
            _ACTIVE[id(self)] = self
        return self

    def _require_active(self):
        if not self.active or _ACTIVE.get(id(self)) is not self or os.getpid() != self.entered_pid:
            raise RuntimeError('Active ownership guard required in its original process')

    def track(self, arena):
        self._require_active()
        address, size = _bounds(arena, unloaded=True)
        if id(arena) in self._tracked:
            raise RuntimeError('Arena was already tracked')
        self._tracked[id(arena)] = dict(arena=arena, address=address, bytes=size, released=False)
        return arena

    def confirm_released(self, arena):
        """Call immediately after successful close, before new VMA allocations."""
        self._require_active()
        record = self._tracked.get(id(arena))
        if record is None or record['arena'] is not arena:
            raise RuntimeError('Cannot confirm an untracked arena')
        if not record['released']:
            if not arena.closed or arena.registered or not arena.mm.closed:
                raise RuntimeError('Arena cleanup is incomplete; ownership guard remains active')
            if _overlaps(record['address'], record['bytes']):
                raise RuntimeError('Arena address remains mapped; ownership guard remains active')
            record['released'] = True

    def release(self):
        self._require_active()
        for record in self._tracked.values():
            self.confirm_released(record['arena'])
        with _LOCK:
            del _ACTIVE[id(self)]
            self.active = False
            self._released = True

    def receipt(self):
        return dict(active=self.active, entered_pid=self.entered_pid,
                    blocked_events=self._blocked_events,
                    tracked_arenas=len(self._tracked),
                    released_arenas=sum(int(r['released']) for r in self._tracked.values()),
                    released=self._released, audit_hook_verified=_HOOK_INSTALLED)


def prewarm_cpu_block():
    """Import lazy dependencies and exercise the exact CPU block helper first."""
    if _ACTIVE:
        raise RuntimeError('Prewarm must precede arena ownership')
    import numpy as np
    import torch
    import dgl
    from candidates.ukl_sage_compact_v1.sampler import block_from_edges
    before = torch.cuda.is_initialized()
    if before:
        raise RuntimeError('CPU prewarm must precede CUDA initialization')
    targets = np.asarray([2, 7], dtype=np.int64)
    inputs = ((np.asarray([9, 2, 9], np.int64), np.asarray([2, 7, 2], np.int64),
               np.asarray([2**33, 2**33 + 1, 2**33 + 2], np.int64)),
              (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)))
    for sources, destinations, eids in inputs:
        block = block_from_edges(targets, sources, destinations, eids)
        if (str(block.device) != 'cpu' or block.num_dst_nodes() != 2
                or block.num_edges() != len(eids)
                or not np.array_equal(block.dstdata[dgl.NID].numpy(), targets)
                or not np.array_equal(block.edata[dgl.EID].numpy(), eids)):
            raise RuntimeError('CPU block prewarm failed')
        block.edges(order='eid')
        del block
    after = torch.cuda.is_initialized()
    if after:
        raise RuntimeError('CPU block prewarm unexpectedly initialized CUDA')
    return dict(passed=True, pid=os.getpid(), torch_version=torch.__version__,
                dgl_version=dgl.__version__, cpu_blocks=len(inputs), cpu_only=True,
                cuda_initialized_before=before, cuda_initialized_after=after,
                before_ownership=True)
