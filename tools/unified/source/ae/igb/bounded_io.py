"""Integer sources: bounded pread, validation and direct final-buffer conversion.

No full mmap, astype or full-array temporary. NPY files are immutable inputs;
the open descriptor's identity is checked before/after each traversal.
"""
import hashlib
import math
import os
import stat
from pathlib import Path
import numpy as np
import torch

DEFAULT_CHUNK = 4 * 1024**2


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class IntegerSource:
    def __init__(self, value):
        self.path = None
        self.array = None
        self.stats = dict(max_read_bytes=0, max_copy_elements=0, traversals=0)
        if isinstance(value, np.memmap):
            # Bundle np.load(mmap_mode='r') arrays must not fault every source
            # page into this process while converting to the final buffer.
            disk = IntegerSource(value.filename)
            if disk.shape != value.shape or disk.dtype != value.dtype or disk.offset != value.offset or not value.flags.c_contiguous:
                raise ValueError('only a complete C-order NPY memmap is accepted')
            self.__dict__.update(disk.__dict__)
            return
        if isinstance(value, (str, Path)):
            self.path = Path(value).resolve()
            with self.path.open('rb') as f:
                s = os.fstat(f.fileno())
                if not stat.S_ISREG(s.st_mode):
                    raise ValueError('regular NPY source required')
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError('NPY v1/v2 required')
                self.offset = f.tell()
                self.signature = identity(s)
            if order:
                raise ValueError('C-order NPY required')
            self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        else:
            if torch.is_tensor(value):
                if value.device.type != 'cpu':
                    raise ValueError('CPU input required')
                value = value.detach().numpy()
            self.array = np.asarray(value)
            if not self.array.flags.c_contiguous:
                raise ValueError('contiguous metadata source required')
            self.shape, self.dtype = self.array.shape, self.array.dtype
        if self.dtype.kind not in 'iu' or self.dtype.itemsize > 8:
            raise ValueError('integer source of at most 64 bits required')
        self.size = math.prod(self.shape)
        self.ndim = len(self.shape)
        if self.path and self.signature[2] != self.offset + self.size * self.dtype.itemsize:
            raise ValueError('NPY file size differs from header')

    def chunks(self, chunk_bytes=DEFAULT_CHUNK):
        if not 8 <= chunk_bytes <= 64 * 1024**2:
            raise ValueError('chunk budget must be in [8 bytes, 64 MiB]')
        count = max(1, chunk_bytes // max(8, self.dtype.itemsize))
        self.stats['traversals'] += 1
        if self.path is None:
            flat = self.array.reshape(-1)
            for at in range(0, self.size, count):
                yield at, flat[at:at+count]
            return
        with self.path.open('rb', buffering=0) as f:
            if identity(os.fstat(f.fileno())) != self.signature:
                raise ValueError('NPY source changed before read')
            for at in range(0, self.size, count):
                size = min(count, self.size-at) * self.dtype.itemsize
                raw = os.pread(f.fileno(), size, self.offset + at*self.dtype.itemsize)
                if len(raw) != size:
                    raise ValueError('short NPY read')
                self.stats['max_read_bytes'] = max(self.stats['max_read_bytes'], size)
                yield at, np.frombuffer(raw, dtype=self.dtype)
                # Drop the previous buffer before pread allocates the next one.
                del raw
            if identity(os.fstat(f.fileno())) != self.signature or identity(self.path.stat()) != self.signature:
                raise ValueError('NPY source changed during read')

    def validate(self, limit=2**63, monotone=False, terminal=None, chunk_bytes=DEFAULT_CHUNK):
        previous = None
        for at, part in self.chunks(chunk_bytes):
            if not part.size:
                continue
            lo, hi = int(part.min()), int(part.max())
            if lo < 0 or hi >= limit:
                raise ValueError('integer source out of range')
            if monotone:
                if (at == 0 and int(part[0]) != 0) or (previous is not None and int(part[0]) < previous):
                    raise ValueError('invalid CSC offset boundary')
                if np.any(part[1:] < part[:-1]):
                    raise ValueError('nonmonotone CSC offsets')
                previous = int(part[-1])
        if terminal is not None and previous != terminal:
            raise ValueError('invalid CSC terminal offset')

    def copy_to(self, target, chunk_bytes=DEFAULT_CHUNK, limit=2**63, monotone=False, terminal=None):
        dest = target.numpy().reshape(-1)
        if dest.size != self.size:
            raise ValueError('destination length mismatch')
        # Revalidate while copying as well: a changed in-memory source cannot
        # silently truncate integers after the separate pre-allocation pass.
        h = hashlib.sha256()
        previous = None
        for at, part in self.chunks(chunk_bytes):
            if part.size and (int(part.min()) < 0 or int(part.max()) >= limit):
                raise ValueError('source changed or conversion overflow')
            if monotone and part.size:
                if (at == 0 and int(part[0]) != 0) or (previous is not None and int(part[0]) < previous) or np.any(part[1:] < part[:-1]):
                    raise ValueError('CSC changed during conversion')
                previous = int(part[-1])
            np.copyto(dest[at:at+part.size], part, casting='unsafe')
            h.update(memoryview(part).cast('B'))
            self.stats['max_copy_elements'] = max(self.stats['max_copy_elements'], int(part.size))
            del part
        if terminal is not None and previous != terminal:
            raise ValueError('CSC terminal changed during conversion')
        return h.hexdigest()


def source(value):
    return value if isinstance(value, IntegerSource) else IntegerSource(value)


def load_csc(indptr, indices, eids, n, host_cap, chunk_bytes=DEFAULT_CHUNK, expected_sha256=None):
    """Load an already-built CSC directly into its final DGL CPU buffers.

    CSC generation, normalization and EID-permutation provenance belong to the
    builder. This reader checks shapes/ranges/order and optional payload SHA;
    it does not infer provenance or repair the rejected legacy IG CSC.
    Passing eids=None defines EIDs in CSC order and generates them in chunks.
    Explicit EIDs must be a builder-verified permutation of [0,E).
    """
    import dgl
    if n <= 0:
        raise ValueError('positive N required')
    sources = [source(indptr), source(indices), None if eids is None else source(eids)]
    p, i, e = sources
    if p.shape != (n+1,) or i.ndim != 1 or (e is not None and e.shape != i.shape):
        raise ValueError('CSC shape mismatch')
    sizes = [p.size, i.size, i.size]
    logical = sum(sizes)*8
    if logical > host_cap:
        raise ValueError('CSC final allocation exceeds explicit host cap')
    p.validate(monotone=True, terminal=i.size, chunk_bytes=chunk_bytes)
    i.validate(limit=n, chunk_bytes=chunk_bytes)
    if e is not None:
        e.validate(limit=max(1, i.size), chunk_bytes=chunk_bytes)
    buffers, hashes = [], []
    for pos, src in enumerate(sources):
        target = torch.empty(sizes[pos], dtype=torch.int64)
        if src is not None:
            h = src.copy_to(target, chunk_bytes, limit=(2**63 if pos == 0 else n if pos == 1 else max(1, i.size)),
                            monotone=pos == 0, terminal=i.size if pos == 0 else None)
        else:
            hsh = hashlib.sha256()
            for at in range(0, i.size, chunk_bytes//8):
                part = np.arange(at, min(i.size, at+chunk_bytes//8), dtype='i8')
                np.copyto(target.numpy()[at:at+len(part)], part)
                hsh.update(memoryview(part).cast('B'))
            h = hsh.hexdigest()
        if expected_sha256 is not None and h != expected_sha256[pos]:
            raise ValueError('CSC payload SHA mismatch')
        buffers.append(target); hashes.append(h)
    graph = dgl.graph(('csc', tuple(buffers)), num_nodes=n, idtype=torch.int64)
    graph = graph.formats('csc')
    actual = graph.adj_tensors('csc')
    if any(a.data_ptr() != b.data_ptr() for a, b in zip(actual, buffers)):
        raise RuntimeError('DGL copied CSC: zero-copy construction required')
    graph.pin_memory_()
    if not graph.is_pinned() or any(a.data_ptr() != b.data_ptr() for a, b in zip(graph.adj_tensors('csc'), buffers)):
        raise RuntimeError('DGL CSC pinning changed storage')
    report = dict(host_final_csc_bytes=logical, csc_pointer_identity=True,
                  temporary_chunk_budget=chunk_bytes, payload_sha256=hashes,
                  sources=[None if s is None else dict(s.stats) for s in sources],
                  generated_csc_order_eids=e is None, explicit_eid_permutation_check=False)
    return graph, report
