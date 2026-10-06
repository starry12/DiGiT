"""Bounded direct reads into owned anonymous memory; no file-cache fallback.

Source offsets are page aligned. A range may end in a partial page only at the
verified end of its source file. Digests cover logical payload bytes, excluding
explicitly zeroed anonymous padding; registration always targets owned pages.
"""
import ctypes as C
import hashlib
import mmap
import os
from pathlib import Path
import stat

PAGE = 4096
CHUNK = 1024 * 1024
MAX_BYTES = 256 * 1024**3
DEFAULT_MAX_BYTES = 8 * 1024**3
RESERVE = 64 * 1024**3
HEADROOM = 8 * 1024**3
RETAINED = []


def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _expected_identity(value):
    if isinstance(value, dict):
        value = tuple(value[k] for k in
                      ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'))
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise ValueError('Expected device/inode/size/mtime/ctime identity')
    if any(type(x) is not int or x < 0 for x in value):
        raise ValueError('Invalid source identity fields')
    return tuple(value)


def admission(remaining_bytes):
    """Reserve only pages still to fault, plus runtime and host headroom."""
    if type(remaining_bytes) is not int or not 0 <= remaining_bytes <= MAX_BYTES:
        raise RuntimeError('Remaining allocation outside bounded admission')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
            break
    required = remaining_bytes + HEADROOM + RESERVE
    if available is None or available < required:
        raise RuntimeError('Host memory admission refused: available=%s required=%s'
                           % (available, required))


def assert_anonymous(address, size):
    end = address + size
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        lo, hi = (int(x, 16) for x in fields[0].split('-'))
        if lo <= address < hi:
            if (hi < end or fields[1] != 'rw-p' or fields[3] != '00:00'
                    or fields[4] != '0' or len(fields) > 5):
                raise RuntimeError('Only private anonymous writable VMAs may be registered')
            return
    raise RuntimeError('Anonymous allocation not found in process maps')


class Arena:
    """Own one anonymous VMA with page-aligned array starts and no adoption.

    specs maps names to {path, offset, length, identity?, sha256?}. Offsets must
    be page aligned; an unaligned logical length must reach the source EOF.
    Each array gets its own page-rounded allocation and zero trailing padding.
    The default allocation cap remains 8 GiB; callers must explicitly supply
    their independently admitted cap to use more, up to a hard 256 GiB ceiling.
    admission_check(remaining_bytes) may also raise for a cooperative stop.
    progress(name, logical_done, logical_length, physical_done, physical_total)
    runs after each successful direct read and padding initialization. Physical
    progress counts pages faulted, while loaded_payload_bytes excludes padding.
    """

    def __init__(self, specs, admission_check=None, *, max_bytes=DEFAULT_MAX_BYTES):
        self.specs = {}
        self.offsets = {}
        self.lengths = {}
        self.allocated_lengths = {}
        self.padding_bytes = {}
        self.digests = {}
        self.source_identities = {}
        self.size = 0
        self.payload_size = 0
        self.total_padding_bytes = 0
        self.loaded_bytes = 0
        self.loaded_payload_bytes = 0
        self.mm = None
        self.address = None
        self.registered = False
        self.loaded = False
        self.closed = False
        self._admission_check = admission if admission_check is None else admission_check
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError('Explicit arena cap must be a positive integer at most 256 GiB')
        self.max_bytes = max_bytes
        if not isinstance(specs, dict) or not specs:
            raise ValueError('At least one named source range required')
        if mmap.PAGESIZE != PAGE or not hasattr(os, 'O_DIRECT') or not hasattr(os, 'preadv'):
            raise RuntimeError('4096-byte pages and Linux O_DIRECT/preadv required')
        for name, spec in specs.items():
            if not isinstance(name, str) or not name or not isinstance(spec, dict):
                raise ValueError('Expected named source-range dictionaries')
            path = Path(spec['path'])
            offset = spec.get('offset', 0)
            length = spec['length']
            if (type(offset) is not int or type(length) is not int
                    or offset < 0 or length <= 0 or offset % PAGE):
                raise ValueError('Source offset must be page aligned and length a positive integer')
            expected = (_expected_identity(spec['identity'])
                        if spec.get('identity') is not None else None)
            digest = spec.get('sha256')
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Expected lowercase SHA256 for requested range')
            if not path.is_absolute() or path.resolve(strict=True) != path:
                raise ValueError('Absolute source path without symlinks required')
            source = path.lstat()
            if not stat.S_ISREG(source.st_mode):
                raise ValueError('Only regular source files may be read')
            actual = identity(source)
            if expected is not None and actual != expected:
                raise RuntimeError('Source identity differs from declared identity')
            if offset + length > source.st_size:
                raise ValueError('Requested range exceeds source file')
            if length % PAGE and offset + length != source.st_size:
                raise ValueError('A partial logical page is permitted only at source EOF')
            allocated = ((length + PAGE - 1) // PAGE) * PAGE
            self.offsets[name] = self.size
            self.lengths[name] = length
            self.allocated_lengths[name] = allocated
            self.padding_bytes[name] = allocated - length
            self.size += allocated
            self.payload_size += length
            self.total_padding_bytes += allocated - length
            self.specs[name] = dict(path=path, offset=offset, length=length,
                                    identity=actual, sha256=digest)
        if not 0 < self.size <= self.max_bytes:
            raise RuntimeError('Page-rounded arena exceeds the explicit allocation cap')
        self._admission_check(self.remaining_bytes)
        self.mm = mmap.mmap(-1, self.size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            self.address = C.addressof(C.c_char.from_buffer(self.mm))
            if self.address % PAGE:
                raise RuntimeError('Anonymous allocation is not page aligned')
            assert_anonymous(self.address, self.size)
        except BaseException:
            self.close()
            raise

    @property
    def remaining_bytes(self):
        return self.size - self.loaded_bytes

    def check_admission(self):
        self._admission_check(self.remaining_bytes)

    def load(self, progress=None):
        if self.closed or self.loaded or self.registered:
            raise RuntimeError('Invalid load lifecycle')
        try:
            for name, spec in self.specs.items():
                path = spec['path']
                self.check_admission()
                # A failed direct open/read is fatal; no buffered-read fallback.
                fd = os.open(str(path), os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW | os.O_CLOEXEC)
                try:
                    before = os.fstat(fd)
                    if (not stat.S_ISREG(before.st_mode)
                            or identity(before) != spec['identity']
                            or identity(path.lstat()) != spec['identity']):
                        raise RuntimeError('Source changed before direct load')
                    digest = hashlib.sha256()
                    done = 0
                    while done < spec['length']:
                        self.check_admission()
                        logical_count = min(CHUNK, spec['length'] - done)
                        count = ((logical_count + PAGE - 1) // PAGE) * PAGE
                        start = self.offsets[name] + done
                        target = memoryview(self.mm)[start:start + count]
                        try:
                            got = os.preadv(fd, [target], spec['offset'] + done)
                            eof_tail = (logical_count != count
                                        and done + logical_count == spec['length']
                                        and spec['offset'] + spec['length'] == before.st_size)
                            if got != logical_count or (got != count and not eof_tail):
                                raise IOError('Invalid O_DIRECT read: requested %d logical %d got %d'
                                              % (count, logical_count, got))
                            if eof_tail:
                                target[logical_count:count] = bytes(count - logical_count)
                            payload = target[:logical_count]
                            try:
                                digest.update(payload)
                            finally:
                                payload.release()
                        finally:
                            target.release()
                        done += logical_count
                        self.loaded_payload_bytes += logical_count
                        self.loaded_bytes += count
                        if progress:
                            progress(name, done, spec['length'], self.loaded_bytes, self.size)
                    if (identity(os.fstat(fd)) != spec['identity']
                            or identity(path.lstat()) != spec['identity']):
                        raise RuntimeError('Source changed during direct load')
                    actual = digest.hexdigest()
                    if spec['sha256'] is not None and actual != spec['sha256']:
                        raise RuntimeError('Requested range SHA256 mismatch')
                    self.digests[name] = actual
                    self.source_identities[name] = spec['identity']
                finally:
                    os.close(fd)
            # An earlier array must not change while a later array is loading.
            for spec in self.specs.values():
                if identity(spec['path'].lstat()) != spec['identity']:
                    raise RuntimeError('Source changed during multi-array load')
            self.check_admission()
            self.loaded = True
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        if self.registered:
            raise RuntimeError('Cannot release memory still registered with CUDA')
        if self.mm is not None:
            # Exported views cause BufferError; do not claim a release in that case.
            self.mm.close()
        self.closed = True


class CudaAPI:
    def __init__(self):
        self.rt = C.CDLL('/usr/local/cuda/lib64/libcudart.so')
        for name, args in {
                'cudaHostRegister': [C.c_void_p, C.c_size_t, C.c_uint],
                'cudaHostGetDevicePointer': [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint],
                'cudaHostUnregister': [C.c_void_p], 'cudaDeviceSynchronize': []}.items():
            fn = getattr(self.rt, name)
            fn.argtypes = args
            fn.restype = C.c_int

    def call(self, name, *args):
        rc = getattr(self.rt, name)(*args)
        if rc:
            raise RuntimeError(name + ' failed: ' + str(rc))

    def register(self, address, size):
        self.call('cudaHostRegister', address, size, 2)

    def device_pointer(self, address):
        p = C.c_void_p()
        self.call('cudaHostGetDevicePointer', C.byref(p), address, 0)
        return p.value

    def synchronize(self):
        self.call('cudaDeviceSynchronize')

    def unregister(self, address):
        self.call('cudaHostUnregister', address)


class Registration:
    def __init__(self, arena, api=None):
        if type(arena) is not Arena or arena.closed or not arena.loaded or arena.registered:
            raise RuntimeError('Loaded owned Arena required')
        arena.check_admission()  # Loaded arena has zero remaining pages to fault.
        assert_anonymous(arena.address, arena.size)
        self.arena = arena
        self.api = CudaAPI() if api is None else api
        self.closed = False
        self.api.register(arena.address, arena.size)
        arena.registered = True
        RETAINED.append(self)
        try:
            base = self.api.device_pointer(arena.address)
            if not base:
                raise RuntimeError('Null CUDA device pointer')
            self.pointers = {k: base + offset for k, offset in arena.offsets.items()}
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        # Preserve strong ownership after sync/unregister failure; never unmap.
        self.api.synchronize()
        self.api.unregister(self.arena.address)
        self.arena.registered = False
        self.closed = True
        RETAINED.remove(self)
