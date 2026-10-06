"""Tiny CPU-only check of all eight EOF pages through the real arena loader."""
import hashlib
import os
from pathlib import Path

from .memory import Arena, PAGE, identity


def expected_tails(specs):
    if len(specs) != 8:
        raise ValueError('Exactly eight source arrays are required')
    tails = {}
    for name, spec in specs.items():
        path = Path(spec['path'])
        st = path.stat()
        if path.is_symlink() or list(identity(st)) != list(spec['identity']):
            raise RuntimeError('Tail source identity changed: '+name)
        offset = ((st.st_size-1)//PAGE)*PAGE
        if offset < 0:
            raise ValueError('Empty source')
        tails[name] = dict(path=str(path), offset=offset,
                           length=st.st_size-offset, identity=list(identity(st)))
    return tails


def run_tail_probe(specs):
    """Read <=32 KiB directly plus <=32 KiB buffered reference; never use CUDA.

    The buffered reference is an independent tiny comparison, not a fallback.
    A direct-I/O error remains fatal. Source identity is checked around reads.
    """
    tails = expected_tails(specs)
    references = {}
    for name, spec in tails.items():
        fd = os.open(spec['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if list(identity(os.fstat(fd))) != spec['identity']:
                raise RuntimeError('Tail reference source changed')
            value = os.pread(fd, spec['length'], spec['offset'])
            if len(value) != spec['length'] or list(identity(os.fstat(fd))) != spec['identity']:
                raise RuntimeError('Short or changed tail reference')
            references[name] = value
        finally:
            os.close(fd)
    arena = Arena(tails, max_bytes=8*PAGE)
    try:
        arena.load()
        arrays = {}
        for name, spec in tails.items():
            offset = arena.offsets[name]
            length = spec['length']
            value = arena.mm[offset:offset+length]
            padding = arena.mm[offset+length:offset+arena.allocated_lengths[name]]
            if value != references[name] or any(padding):
                raise RuntimeError('Direct EOF bytes or zero padding mismatch: '+name)
            if list(identity(Path(spec['path']).stat())) != spec['identity']:
                raise RuntimeError('Tail source changed after load')
            arrays[name] = dict(spec, allocated_bytes=arena.allocated_lengths[name],
                source_sha256=arena.digests[name],
                reference_sha256=hashlib.sha256(references[name]).hexdigest(), padding_zero=True)
        return dict(read_bytes=arena.loaded_payload_bytes,
            requested_read_bytes=arena.loaded_bytes, reference_bytes=sum(map(len, references.values())),
            direct_io=True, gpu_called=False, raw_ssd_access=False, arrays=arrays)
    finally:
        arena.close()
