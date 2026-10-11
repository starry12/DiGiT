"""Select an observable ancestor only for host-monitor qualification."""
from pathlib import Path


def monitor_target(controller_cgroup, cgroot='/sys/fs/cgroup'):
    root = Path(cgroot).resolve()
    if not controller_cgroup.startswith('/'):
        raise ValueError('Controller cgroup must be absolute')
    current = (root / controller_cgroup.lstrip('/')).resolve()
    current.relative_to(root)
    required = ('io.stat', 'memory.current', 'memory.stat', 'memory.events')
    while True:
        if all((current / name).is_file() for name in required):
            return '/' + str(current.relative_to(root)) if current != root else '/'
        if current == root:
            raise RuntimeError('No observable ancestor cgroup for monitor qualification')
        current = current.parent
