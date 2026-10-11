"""Bounded O_DIRECT array I/O; no file-backed CUDA registration or cache flooding."""
import hashlib
import mmap
import os
from pathlib import Path
import numpy as np
from .common import require

CHUNK = 8*2**20


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def direct_chunks(path, expected=None, check=lambda: None):
    path = Path(path); before = identity(path)
    require(not path.is_symlink(), 'Symlink input')
    if expected:
        require(before == expected['identity'] and before[2] == expected['bytes'], 'Input identity changed')
    h = hashlib.sha256(); total = 0
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
    buffer = mmap.mmap(-1, CHUNK)
    view = memoryview(buffer)
    try:
        while total < before[2]:
            check(); got = os.readv(fd, [view])
            require(got > 0, 'Short direct read')
            h.update(view[:got]); total += got
            yield view[:got]
        require(total == before[2] and identity(path) == before, 'Input mutated during direct read')
        if expected: require(h.hexdigest() == expected['sha256'], 'Input digest mismatch')
    finally:
        view.release(); buffer.close(); os.close(fd)


def read_array(receipt, dtype=np.int64, check=lambda: None):
    dtype = np.dtype(dtype); require(receipt['bytes'] % dtype.itemsize == 0, 'Array extent')
    result = np.empty(receipt['bytes']//dtype.itemsize, dtype)
    cursor = 0
    for chunk in direct_chunks(receipt['path'], receipt, check):
        part = np.frombuffer(chunk, dtype); result[cursor:cursor+len(part)] = part
        cursor += len(part); del part; chunk.release()
    return result


def selected_scores(path, nodes, selected, check=lambda: None, expected_sha=None, header_bytes=0):
    require(0 <= header_bytes < CHUNK and header_bytes % 8 == 0, 'Rank header alignment')
    require(Path(path).stat().st_size == nodes*8+header_bytes, 'RevPR rank extent')
    result = np.empty(len(selected), np.float64); start = 0
    evidence = dict(path=str(path), bytes=nodes*8+header_bytes, identity=identity(path))
    if expected_sha: evidence['sha256'] = expected_sha
    for chunk in direct_chunks(path, evidence if expected_sha else None, check):
        values = np.frombuffer(chunk, np.float64, offset=header_bytes)
        header_bytes=0
        lo, hi = np.searchsorted(selected, [start, start+len(values)])
        result[lo:hi] = values[selected[lo:hi]-start]
        start += len(values); del values; chunk.release()
    return result, evidence


def save_array(path, array, check=lambda: None):
    from candidates.ukl_runtime_prepare_v11r1.data import direct_file
    def chunks():
        for lo in range(0, len(array), CHUNK//array.dtype.itemsize):
            yield array[lo:lo+CHUNK//array.dtype.itemsize]
    return direct_file(Path(path), chunks(), check, 256*2**20)
