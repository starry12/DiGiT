"""Bounded native CSC sampling over the exact v9 owned anonymous arena.

This adapter never adopts external arrays and never creates registrations.
Sampling and block assembly are acceptance interfaces, not timing claims.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from candidates.ukl_real_multi_v9.memory import Arena, Registration, assert_anonymous
from candidates.ukl_sparse_overlay_v3.native import BUILD, View, Out, outputs


MAX_SEEDS = 36864                 # 1024 * (1 + 5) * (1 + 5)
MAX_OUTPUT_EDGES = 1048576
MAX_SCRATCH_BYTES = 128 * 1024**2
CHECK_CHUNK = 65536
RETAINED = []                     # Failed GPU operations retain every owner.
_NAMES = {'ptr': 'indptr', 'idx': 'indices', 'gptr': 'group_ptr',
          'gidx': 'group_ids', 'covered': 'covered', 'bases': 'bases',
          'primary': 'primary', 'members': 'members'}
_ORDER = ('ptr', 'gptr', 'bases', 'primary', 'covered', 'idx', 'gidx', 'members')
_FIELDS = ('nodes', 'edges', 'groups', 'units', 'storage_rows', 'origin', 'unit_origin')
_I64_MAX = 2**63 - 1


class Graph:
    """Own readonly exported NumPy views; the caller owns the v9 Arena.

    Only scalar metadata endpoints are inspected at construction. Referenced
    owners and outputs are checked per call in bounded chunks. Empty payloads
    are deliberately unsupported because v9 Arena requires positive ranges.
    """

    def __init__(self, arena, metadata):
        if type(arena) is not Arena or arena.closed or not arena.loaded:
            raise ValueError('An exact loaded v9 Arena is required')
        if not isinstance(metadata, dict) or set(metadata) != set(_FIELDS):
            raise ValueError('Exact graph metadata fields required')
        if any(type(metadata[k]) is not int for k in _FIELDS):
            raise ValueError('Graph dimensions and origins must be integers')
        n, e, g, u, rows, origin, unit_origin = (metadata[k] for k in _FIELDS)
        if not (0 < n < 2**31 and 0 < e <= _I64_MAX and 0 < g < 2**31
                and 0 < u <= e // 2 and 0 < rows <= (_I64_MAX // 512)
                and 0 <= origin <= _I64_MAX - e and unit_origin == 0):
            raise ValueError('Unsupported graph metadata bounds')
        if set(arena.offsets) != set(_NAMES.values()):
            raise ValueError('Arena must contain exactly the eight graph payloads')
        assert_anonymous(arena.address, arena.size)
        self.arena = arena
        self._owner = arena
        self.metadata = MappingProxyType(dict(metadata))
        self.nodes, self.edges, self.groups, self.units = n, e, g, u
        self.storage_rows, self.origin, self.unit_origin = rows, origin, unit_origin
        self.closed = False
        self._arrays = {}
        self._expected = {}
        specs = {'ptr': (np.int64, (n + 1,)), 'idx': (np.int32, (e,)),
                 'gptr': (np.int64, (n + 1,)), 'gidx': (np.int32, (u,)),
                 'covered': (np.int64, (2 * u,)), 'bases': (np.int64, (g,)),
                 'primary': (np.int64, (n,)), 'members': (np.int32, (g, 2))}
        try:
            for name, (dtype, shape) in specs.items():
                source = _NAMES[name]
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                offset = arena.offsets[source]
                if (arena.lengths[source] != size or offset < 0
                        or offset + size > arena.size or offset % 4096):
                    raise ValueError('Graph payload width/shape mismatch: ' + source)
                # frombuffer preserves an actual exported memoryview. An
                # external borrower consequently prevents Arena.close().
                parent = memoryview(arena.mm).toreadonly()
                segment = parent[offset:offset + size]
                try:
                    value = np.frombuffer(segment, dtype=dtype).reshape(shape)
                finally:
                    segment.release()
                    parent.release()
                self._arrays[name] = value
                self._expected[name] = (id(value), np.dtype(dtype), shape,
                                        arena.address + offset)
            self.arrays = MappingProxyType(self._arrays)
            self._check_owned()
            if (int(self.arrays['ptr'][0]) != origin
                    or int(self.arrays['ptr'][-1]) != origin + e
                    or int(self.arrays['gptr'][0]) != 0
                    or int(self.arrays['gptr'][-1]) != u):
                raise ValueError('Graph pointer endpoints differ from metadata')
        except BaseException:
            self._arrays.clear()
            self.closed = True
            raise

    def _check_owned(self):
        if self.arena is not self._owner or type(self.arena) is not Arena:
            raise ValueError('Graph arena ownership changed')
        if self.closed or self.arena.closed or not self.arena.loaded:
            raise RuntimeError('Graph or arena is closed')
        if any(getattr(self, key) != self.metadata[key] for key in _FIELDS):
            raise ValueError('Graph metadata changed after view construction')
        if set(self._arrays) != set(_NAMES):
            raise ValueError('Graph view ownership changed')
        for name, value in self._arrays.items():
            ident, dtype, shape, address = self._expected[name]
            if (type(value) is not np.ndarray or id(value) != ident
                    or value.dtype != dtype or value.shape != shape
                    or not value.flags.c_contiguous or value.flags.writeable
                    or value.flags.owndata or value.ctypes.data != address):
                raise ValueError('Graph view ownership changed: ' + name)

    def close(self):
        if not self.closed:
            self._arrays.clear()
            self.closed = True


def _binary(gpu):
    binary = BUILD / ('libcompact_cuda.so' if gpu else 'libcompact_cpu.so')
    manifest = Path(__file__).resolve().parents[1] / 'ukl_runtime_prepare_v6/dependencies.json'
    expected = json.loads(manifest.read_text())
    key = str(binary.relative_to(Path(__file__).resolve().parents[2]))
    digest = hashlib.sha256()
    with binary.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected.get(key):
        raise RuntimeError('Native binary differs from the v6 dependency manifest')
    return binary


class Native:
    def __init__(self, graph, registration=None):
        if type(graph) is not Graph:
            raise ValueError('Owned Graph required; external arrays are forbidden')
        graph._check_owned()
        self.graph = graph
        self.registration = registration
        self.gpu = registration is not None
        self.closed = False
        self.retained = False
        self._registration_pointers = None
        self._check_registration()
        self.lib = C.CDLL(str(_binary(self.gpu)))
        self.fn = self.lib.sample_cuda if self.gpu else self.lib.sample_cpu
        self.fn.argtypes = [View, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_uint64, Out]
        self.fn.restype = C.c_int
        if self.gpu:
            for name, arguments in {
                    'cudaMalloc': [C.POINTER(C.c_void_p), C.c_size_t],
                    'cudaFree': [C.c_void_p],
                    'cudaMemcpy': [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
                    'cudaMemset': [C.c_void_p, C.c_int, C.c_size_t]}.items():
                function = getattr(registration.api.rt, name)
                function.argtypes = arguments
                function.restype = C.c_int

    def _check_registration(self):
        reg = self.registration
        if reg is not None:
            if (type(reg) is not Registration or reg.arena is not self.graph.arena
                    or reg.closed or not reg.arena.registered
                    or set(reg.pointers) != set(_NAMES.values())):
                raise ValueError('Existing registration must own this exact v9 Arena')
            offsets = self.graph.arena.offsets
            bases = {int(reg.pointers[k]) - offsets[k] for k in offsets}
            if len(bases) != 1 or next(iter(bases)) <= 0:
                raise ValueError('Registration device pointers lost arena ownership')
            if self._registration_pointers is None:
                # A common offset alone cannot prove ownership: all pointers
                # could have been shifted together. Attest to this registered
                # host base once, then reject any later mapping mutation.
                base = reg.api.device_pointer(reg.arena.address)
                expected = {key: base + offset for key, offset in offsets.items()} if base else {}
                if reg.pointers != expected:
                    raise ValueError('Registration pointers do not identify the owned arena')
                self._registration_pointers = dict(expected)
            elif reg.pointers != self._registration_pointers:
                raise ValueError('Registration device pointer mapping changed')

    def _view(self, packed):
        graph = self.graph
        pointers = ({k: self.registration.pointers[_NAMES[k]] for k in _ORDER}
                    if self.gpu else {k: int(graph.arrays[k].ctypes.data) for k in _ORDER})
        if not packed:
            pointers['primary'] = None
        return View(graph.nodes, graph.edges, graph.origin, graph.units,
                    graph.unit_origin, graph.groups, *[pointers[k] for k in _ORDER])

    def _owners(self, seeds, grouped):
        """Check precisely the referenced pointer pairs and covered segments.

        No whole-graph slice, adjacency copy or edge-sized temporary is used.
        Sorted covers bound the native binary search to the owner's CSC row.
        The native routine checks selected group IDs/members before access.
        """
        a, graph = self.graph.arrays, self.graph
        begin = a['ptr'][seeds]
        end = a['ptr'][seeds.astype(np.int64) + 1]
        if np.any(begin < graph.origin) or np.any(end < begin) or np.any(end > graph.origin + graph.edges):
            raise ValueError('Referenced CSC pointer pair outside graph')
        if grouped:
            left = a['gptr'][seeds]
            right = a['gptr'][seeds.astype(np.int64) + 1]
            if (np.any(left < 0) or np.any(right < left)
                    or np.any(right > graph.units) or np.any(right - left > (end - begin) // 2)):
                raise ValueError('Referenced group pointer pair outside graph')
            for lo, hi, start, stop in zip(left, right, begin, end):
                lo, hi = 2 * int(lo), 2 * int(hi)
                previous = int(start) - 1
                for offset in range(lo, hi, CHECK_CHUNK):
                    cover = a['covered'][offset:min(hi, offset + CHECK_CHUNK)]
                    if (int(cover[0]) <= previous or int(cover[-1]) >= int(stop)
                            or np.any(cover[1:] <= cover[:-1])):
                        raise ValueError('Referenced covered EIDs are unsorted or out of owner')
                    previous = int(cover[-1])
        return begin, end

    def sample(self, seeds, fanout, grouped=False, seed=0, packed=True):
        if self.closed or self.retained:
            raise RuntimeError('Native sampler is closed or retained after CUDA failure')
        self.graph._check_owned()
        self._check_registration()
        seeds = np.asarray(seeds)
        if (seeds.dtype != np.int32 or seeds.ndim != 1
                or not 0 < len(seeds) <= MAX_SEEDS or seeds.min() < 0
                or seeds.max() >= self.graph.nodes or type(fanout) is not int
                or not 1 <= fanout <= 32 or type(grouped) is not bool
                or type(packed) is not bool or (grouped and not packed)
                or type(seed) is not int or not 0 <= seed <= 2**64 - 1):
            raise ValueError('Sampling arguments exceed the bounded contract')
        slots = len(seeds) * fanout
        # Simultaneous host/device raw output and input, plus bounded validation
        # temporaries. This conservative budget also applies to the CPU path.
        scratch = 2 * (24 * slots + 12 * len(seeds)) + 32 * slots + 16 * CHECK_CHUNK
        if slots > MAX_OUTPUT_EDGES or scratch > MAX_SCRATCH_BYTES:
            raise ValueError('Per-call sampled outputs/scratch exceed admission')
        seeds = np.array(seeds, dtype=np.int32, order='C', copy=True)
        begin, end = self._owners(seeds, grouped)
        out = outputs(len(seeds), fanout)
        native_view = self._view(packed)
        if self.gpu:
            rc = self._cuda(native_view, seeds, fanout, grouped, seed, out)
        else:
            rc = self.fn(native_view, int(seeds.ctypes.data), len(seeds), fanout,
                         int(grouped), seed, Out(*[int(x.ctypes.data) for x in out.values()]))
        if rc or np.any(out['errors']):
            raise RuntimeError('Native sampling rejected graph or operation: ' + str(rc))
        self._validate_output(seeds, fanout, grouped, packed, begin, end, out)
        return out

    def _cuda(self, native_view, seeds, fanout, grouped, seed, out):
        api = self.registration.api
        allocated = []
        synchronized = False
        retain = False

        def allocate(size):
            pointer = C.c_void_p()
            api.call('cudaMalloc', C.byref(pointer), size)
            if not pointer.value:
                raise RuntimeError('CUDA returned a null allocation')
            allocated.append(pointer)
            return pointer

        def synchronize():
            nonlocal synchronized, retain
            try:
                api.synchronize()
                synchronized = True
            except BaseException:
                retain = True
                raise

        try:
            sp = allocate(seeds.nbytes)
            api.call('cudaMemcpy', sp, int(seeds.ctypes.data), seeds.nbytes, 1)
            op = {key: allocate(value.nbytes) for key, value in out.items()}
            for key, value in out.items():
                api.call('cudaMemset', op[key], 255, value.nbytes)
            rc = self.fn(native_view, sp, len(seeds), fanout, int(grouped), seed,
                         Out(*[pointer.value for pointer in op.values()]))
            synchronize()
            for key, value in out.items():
                synchronized = False
                api.call('cudaMemcpy', int(value.ctypes.data), op[key], value.nbytes, 2)
            return rc
        finally:
            try:
                if not retain and not synchronized:
                    synchronize()
                if not retain:
                    while allocated:
                        api.call('cudaFree', allocated[-1])
                        allocated.pop()
            except BaseException:
                retain = True
                raise
            finally:
                if retain:
                    self.retained = True
                    RETAINED.append({'native': self, 'graph': self.graph,
                                     'registration': self.registration,
                                     'seeds': seeds, 'outputs': out,
                                     'allocations': allocated})

    def _validate_output(self, seeds, fanout, grouped, packed, begin, end, out):
        counts = out['counts']
        if np.any(counts < 0) or np.any(counts > fanout) or np.any(counts > end - begin):
            raise RuntimeError('Native output counts outside sample bounds')
        if not grouped and not np.array_equal(counts, np.minimum(fanout, end - begin)):
            raise RuntimeError('Native baseline output count mismatch')
        if grouped:
            target = np.minimum(fanout, end - begin)
            left = self.graph.arrays['gptr'][seeds]
            right = self.graph.arrays['gptr'][seeds.astype(np.int64) + 1]
            raw = end - begin - 2 * (right - left)
            # A budget may leave exactly one unusable slot when only groups
            # remain. Any larger shortfall means an eligible unit was omitted.
            short_one = ((counts == fanout - 1) & (counts < end - begin - 1)
                         & (raw <= counts) & (right > left))
            if np.any((counts != target) & ~short_one):
                raise RuntimeError('Native grouped output count mismatch')
        # Validation copies scale only with sampled output and remain bounded.
        for first in range(0, len(seeds), max(1, CHECK_CHUNK // fanout)):
            last = min(len(seeds), first + max(1, CHECK_CHUNK // fanout))
            mask = np.arange(fanout)[None, :] < counts[first:last, None]
            fields = {k: out[k].reshape(-1, fanout)[first:last] for k in ('src', 'dst', 'eid', 'rows')}
            src, dst, eid, rows = (fields[k][mask] for k in ('src', 'dst', 'eid', 'rows'))
            starts = np.broadcast_to(begin[first:last, None], mask.shape)[mask]
            stops = np.broadcast_to(end[first:last, None], mask.shape)[mask]
            owners = np.broadcast_to(seeds[first:last, None], mask.shape)[mask]
            if (np.any(src < 0) or np.any(src >= self.graph.nodes)
                    or not np.array_equal(dst, owners) or np.any(eid < starts)
                    or np.any(eid >= stops) or np.any(rows < 0)
                    or np.any(rows >= (self.graph.storage_rows if packed else self.graph.nodes))):
                raise RuntimeError('Native output node/EID/storage row outside graph')
            if not np.array_equal(self.graph.arrays['idx'][eid - self.graph.origin], src):
                raise RuntimeError('Native output EID does not resolve to source')
            expected_rows = self.graph.arrays['primary'][src] if packed else src
            if not grouped and not np.array_equal(expected_rows, rows):
                raise RuntimeError('Native output storage rows differ from declared mode')
            # EIDs are occurrence identities; duplicate sources remain legal.
            ordered = np.sort(np.where(mask, fields['eid'], -1), axis=1)
            if np.any((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])):
                raise RuntimeError('Native output repeats an occurrence for one owner')
            for key, value in fields.items():
                if np.any(value[~mask] != -1):
                    raise RuntimeError('Native wrote beyond declared output count: ' + key)
            if grouped:
                self._validate_group_rows(seeds[first:last], fields, counts[first:last])

    def _validate_group_rows(self, seeds, fields, counts):
        a = self.graph.arrays
        for i, owner in enumerate(seeds):
            amount = int(counts[i])
            for position in range(amount):
                source = int(fields['src'][i, position])
                row = int(fields['rows'][i, position])
                if row == int(a['primary'][source]):
                    continue
                # A packed row must identify the matching member of a group
                # referenced by this owner; scan bounded chunks without maps.
                left, right = int(a['gptr'][owner]), int(a['gptr'][int(owner) + 1])
                found = False
                for start in range(left, right, CHECK_CHUNK):
                    gids = a['gidx'][start:min(right, start + CHECK_CHUNK)]
                    if np.any(gids < 0) or np.any(gids >= self.graph.groups):
                        raise RuntimeError('Referenced group ID outside graph')
                    bases = a['bases'][gids]
                    candidates = np.flatnonzero((bases == row) | (bases == row - 1))
                    for candidate in candidates:
                        gid = int(gids[candidate])
                        slot = row - int(bases[candidate])
                        if int(a['members'][gid, slot]) == source:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise RuntimeError('Packed row does not identify the sampled source')

    def close(self):
        # The caller owns both Registration and Arena, including failure cleanup.
        self.closed = True


def compact(raw, fanout):
    mask = np.arange(fanout)[None, :] < raw['counts'][:, None]
    return {key: raw[key].reshape(-1, fanout)[mask].copy()
            for key in ('src', 'dst', 'eid', 'rows')}


def stable_unique(array):
    array = np.asarray(array)
    _, first = np.unique(array, return_index=True)
    return array[np.sort(first)].copy()


class Sampler:
    def __init__(self, native, grouped=False, seed=0):
        if type(grouped) is not bool or type(seed) is not int or not 0 <= seed < 2**64 - 2:
            raise ValueError('Sampler mode/seed bounds')
        self.native, self.grouped, self.seed = native, grouped, seed

    def layers(self, roots, batch=0, audit=None):
        roots = np.asarray(roots)
        if (roots.ndim != 1 or roots.dtype.kind not in 'iu' or not 0 < len(roots) <= 1024
                or np.any(roots < 0) or np.any(roots >= self.native.graph.nodes)
                or len(np.unique(roots)) != len(roots) or type(batch) is not int or batch < 0
                or self.seed + batch * 3 + 2 > 2**64 - 1):
            raise ValueError('Unique roots, batch and seed must satisfy 1024-root bounds')
        seeds = np.array(roots, dtype=np.int32, copy=True)
        layers = []
        for layer in (2, 1, 0):
            fanout = (10, 5, 5)[layer]
            grouped = self.grouped and layer == 0
            raw = self.native.sample(seeds, fanout, grouped=grouped,
                                     seed=self.seed + batch * 3 + layer, packed=self.grouped)
            if audit is not None:
                audit(layer, seeds.copy(), fanout, grouped, self.grouped, raw)
            result = compact(raw, fanout)
            frontier = stable_unique(np.concatenate((seeds, result['src'])))
            layers.insert(0, (seeds.copy(), frontier.copy(), result))
            seeds = frontier
        return seeds.copy(), layers

    def sample_blocks(self, roots, batch=0, *, layers=None):
        import torch
        import dgl
        from candidates.ukl_sage_compact_v1.sampler import block_from_edges, STORAGE_ROW
        nodes, layers = self.layers(roots, batch) if layers is None else layers
        blocks = []
        for layer, (dst, src, result) in enumerate(layers):
            block = block_from_edges(dst, result['src'], result['dst'], result['eid'])
            if not np.array_equal(block.srcdata[dgl.NID].numpy(), src):
                raise RuntimeError('Block frontier ordering changed')
            if layer == 0:
                addresses = np.array(self.native.graph.arrays['primary'][src]
                                     if self.grouped else src, dtype=np.int64, copy=True)
                _, first = np.unique(result['src'], return_index=True)
                lookup = {int(node): i for i, node in enumerate(src)}
                for i in first:
                    addresses[lookup[int(result['src'][i])]] = result['rows'][i]
                if np.any(addresses < 0) or np.any(addresses >= (self.native.graph.storage_rows
                                                                 if self.grouped else self.native.graph.nodes)):
                    raise RuntimeError('Block storage rows outside graph')
                block.srcdata[STORAGE_ROW] = torch.from_numpy(addresses)
            blocks.append(block)
        return (torch.from_numpy(np.array(nodes, dtype=np.int64, copy=True)),
                torch.from_numpy(np.array(roots, dtype=np.int64, copy=True)), blocks)
