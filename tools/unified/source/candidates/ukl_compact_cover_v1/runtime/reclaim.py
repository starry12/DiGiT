"""Bounded cache advice for identity-checked input ranges, outside training.

An advisory return value is never admission evidence: callers re-read MemFree.
No global drop_caches, cache-filling read, migration, sync, or raw-device access.
"""
import contextlib
import os
import stat
import time
from pathlib import Path

PAGE = 4096
CHUNK = 256 * 2**20
MAX_BYTES = 336 * 2**30 # authenticated compact arena can exceed the old 256 GiB cap
MAX_SECONDS = 120


def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def expected(value):
    if isinstance(value, dict):
        value = tuple(value[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'))
    if not isinstance(value, (list, tuple)) or len(value) != 5 or any(type(v) is not int or v < 0 for v in value):
        raise ValueError('Full source identity required for cache advice')
    return tuple(value)


def recover(sources, needed, memory, checkpoint, emit):
    start = time.monotonic()
    result = dict(kind='targeted_file_cache_reclaim', method='POSIX_FADV_DONTNEED',
                  requested_bytes=0, calls=0, sources=[], before=memory(),
                  max_bytes=MAX_BYTES, max_seconds=MAX_SECONDS, chunk_bytes=CHUNK,
                  global_drop_caches=False, memory_threshold_lowered=False)
    def enough(sample):
        return all(type(sample[n]['MemFree']) is int and sample[n]['MemFree'] >= needed for n in ('0', '1'))
    try:
        if not isinstance(sources, dict) or not sources:
            raise ValueError('Only authenticated arena source ranges are eligible')
        # Validate and hold every descriptor before the first advisory operation.
        with contextlib.ExitStack() as stack:
            opened = []; total = 0; seen = set()
            for name, item in sources.items():
                path = Path(item['path']); off = item.get('offset', 0); length = item['length']
                if not path.is_absolute() or path.resolve(strict=True) != path:
                    raise ValueError('Source must be an absolute non-symlink path')
                if type(off) is not int or off < 0 or off % PAGE or type(length) is not int or length <= 0:
                    raise ValueError('Invalid source range')
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
                stack.callback(os.close, fd)
                st = os.fstat(fd); ident = expected(item['identity'])
                if not stat.S_ISREG(st.st_mode) or identity(st) != ident or identity(path.stat()) != ident:
                    raise RuntimeError('Cache-advice source identity/type changed')
                if off + length > st.st_size:
                    raise ValueError('Source range exceeds EOF')
                # Do not extend requests past the authenticated range, including its tail.
                length = length // PAGE * PAGE
                key = (st.st_dev, st.st_ino, off, length)
                if key in seen:
                    continue
                seen.add(key); total += length
                if total > MAX_BYTES:
                    raise ValueError('Cache-advice source budget exceeded')
                row = dict(name=name, path=str(path), offset=off, length=length,
                           identity=list(ident), requested_bytes=0, calls=0)
                result['sources'].append(row); opened.append((fd, path, ident, row))
            result['after'] = memory()
            emit(dict(result, phase='begin'))
            stop = enough(result['after'])
            for fd, path, ident, row in opened:
                cursor = row['offset']; end = cursor + row['length']
                while cursor < end and not stop:
                    checkpoint()
                    if time.monotonic() - start >= MAX_SECONDS:
                        result['budget_exhausted'] = True; stop = True; break
                    if identity(os.fstat(fd)) != ident or identity(path.stat()) != ident:
                        raise RuntimeError('Cache-advice source changed while reclaiming')
                    size = min(CHUNK, end-cursor)
                    os.posix_fadvise(fd, cursor, size, os.POSIX_FADV_DONTNEED)
                    cursor += size; row['requested_bytes'] += size; row['calls'] += 1
                    result['requested_bytes'] += size; result['calls'] += 1
                    if identity(os.fstat(fd)) != ident or identity(path.stat()) != ident:
                        raise RuntimeError('Cache-advice source changed after advice')
                    result['after'] = memory()
                    stop = enough(result['after'])
                    if stop or result['calls'] % 16 == 0:
                        emit(dict(result, phase='progress', elapsed_seconds=time.monotonic()-start))
                if stop: break
            result['after'] = memory()
            result['memory_sufficient'] = enough(result['after'])
    except BaseException as error:
        result['error'] = repr(error)
        result['elapsed_seconds'] = time.monotonic()-start
        emit(dict(result, phase='error'))
        raise
    result['elapsed_seconds'] = time.monotonic()-start
    emit(dict(result, phase='complete'))
    return result
