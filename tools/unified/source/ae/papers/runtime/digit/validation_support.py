"""Standalone bitmap and resource helpers for runtime artifact validation."""
from pathlib import Path
from collections import OrderedDict
import os,math,resource
MIB=1024**2

def nearest_existing(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def free_bytes(path):
    s = os.statvfs(nearest_existing(path))
    return s.f_bavail * s.f_frsize


def memory_snapshot():
    """Read host MemAvailable and visible cgroup-v2 ancestor headroom."""
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    limits = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            root = Path("/sys/fs/cgroup")
            path = root / line[3:].lstrip("/")
            while path == root or root in path.parents:
                maximum, current = path / "memory.max", path / "memory.current"
                if maximum.is_file() and current.is_file():
                    text = maximum.read_text().strip()
                    if text != "max":
                        limits.append(max(0, int(text) - int(current.read_text())))
                if path == root:
                    break
                path = path.parent
    candidates = limits + ([] if available is None else [available])
    return dict(host_available_bytes=available, visible_cgroup_v2_headroom_bytes=limits,
                effective_available_bytes=min(candidates) if candidates else None,
                note="Read-only snapshot, not a reservation; cgroup-v1 is not inferred")


def check_heap(bound, memory_mib):
    if bound > memory_mib * MIB:
        raise ValueError("buffer/manifest budget exceeded; reduce chunk/batch size")
    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    vm = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    if soft != resource.RLIM_INFINITY and vm + bound > soft:
        raise ValueError("planned buffers do not fit the address-space ceiling")
    available = memory_snapshot()["effective_available_bytes"]
    if available is not None and bound > available * .8:
        raise ValueError("planned buffers exceed 80% of current available memory")


def check_space(path, size):
    if math.ceil(size * 1.2) > free_bytes(path):
        raise ValueError("insufficient free space including 20% headroom")


class Bits:
    """Disposable file-backed bitmap with a fixed 512-KiB page cache."""
    def __init__(self, path, count):
        self.path, self.count = Path(path), count
        if self.path.is_symlink():
            raise ValueError("bitmap symlink")
        self.file = self.path.open("w+b")
        self.file.truncate((count + 7) // 8)
        self.cache = OrderedDict()

    def page(self, index):
        if not 0 <= index < self.count:
            raise ValueError("bitmap ID outside range")
        key = index // 32768
        if key not in self.cache:
            if len(self.cache) >= 128:
                removed, data = self.cache.popitem(last=False)
                self.file.seek(removed * 4096)
                self.file.write(data)
            self.file.seek(key * 4096)
            self.cache[key] = bytearray(self.file.read(min(4096, (self.count + 7) // 8 - key * 4096)))
        self.cache.move_to_end(key)
        return self.cache[key]

    def get(self, index):
        return bool(self.page(index)[(index // 8) % 4096] & (1 << (index % 8)))

    def set(self, index):
        value = self.get(index)
        self.page(index)[(index // 8) % 4096] |= 1 << (index % 8)
        return value

    def close(self):
        self.file.close()
        self.path.unlink()  # Only this validator's owned disposable scratch.


