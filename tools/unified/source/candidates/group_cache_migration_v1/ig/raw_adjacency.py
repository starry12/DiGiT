"""Explicit raw int64 source for the two prepared IG adjacency files.

Retains bounded pread and copies only into the caller's final anonymous buffer.
It never registers a file mapping with CUDA or guesses a format from an error.
"""
import os
import stat
from pathlib import Path
import numpy as np
from bounded_io import IntegerSource, identity, DEFAULT_CHUNK


class RawAdjacencySource(IntegerSource):
    def __init__(self, value, spec):
        if spec.get('format') != 'raw' or spec.get('dtype') != '<i8':
            raise ValueError('Explicit raw little-endian int64 descriptor required')
        path = Path(spec['path'])
        if path.is_symlink():
            raise ValueError('Raw adjacency symlink is not accepted')
        self.path = path.resolve()
        self.array = None
        self.dtype = np.dtype('<i8')
        self.offset = spec['offset']
        length = spec['length']
        if self.offset != 0 or not isinstance(length, int) or length < 0 or length % 8:
            raise ValueError('Invalid raw adjacency offset/length')
        self.size = length // 8
        self.shape, self.ndim = (self.size,), 1
        if list(self.shape) != spec['shape']:
            raise ValueError('Raw adjacency shape differs from descriptor')
        with self.path.open('rb', buffering=0) as f:
            s = os.fstat(f.fileno())
            if not stat.S_ISREG(s.st_mode):
                raise ValueError('Regular raw adjacency file required')
            self.signature = identity(s)
        if self.signature[2] != length or list(self.signature[1:]) != spec['identity'][1:]:
            raise ValueError('Raw adjacency identity/length changed')
        if (not isinstance(value, np.memmap) or value.mode != 'r'
                or Path(value.filename).resolve() != self.path
                or value.dtype != self.dtype or value.shape != self.shape
                or value.offset != self.offset or not value.flags.c_contiguous):
            raise ValueError('Complete read-only raw adjacency mapping required')
        self.expected_sha256 = spec['sha256']
        if (not isinstance(self.expected_sha256, str) or len(self.expected_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.expected_sha256)):
            raise ValueError('Raw adjacency digest required')
        self.stats = dict(max_read_bytes=0, max_copy_elements=0, traversals=0)

    def copy_to(self, target, chunk_bytes=DEFAULT_CHUNK, limit=2**63,
                monotone=False, terminal=None):
        digest = super().copy_to(target, chunk_bytes, limit, monotone, terminal)
        if digest != self.expected_sha256:
            raise ValueError('Raw adjacency payload SHA mismatch')
        return digest
