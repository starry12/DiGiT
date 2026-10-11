"""Resolve the owned phase leaf without dropping the aggregate service limits."""
from pathlib import Path, PurePosixPath


def service_group(cgroup):
    p=PurePosixPath(cgroup)
    if (not cgroup.startswith('/system.slice/digit-cl-training-v10-')
            or p.name not in ('init','data') or not p.parent.name.endswith('.service')
            or len(p.parts)!=4 or '..' in p.parts):
        raise RuntimeError('Exact training service phase cgroup required')
    return str(p.parent)


def verify_leaf(leaf, parent):
    # Parent bounds the sum; matching leaf limits cannot enlarge that allowance.
    for name in ('memory.max','memory.high','memory.swap.max','pids.max','cpu.max','io.max'):
        if (leaf/name).read_text().strip()!=(parent/name).read_text().strip():
            raise RuntimeError('Phase resource limit mismatch: '+name)
    if (leaf/'cgroup.type').read_text().strip()!='domain':
        raise RuntimeError('Phase must remain a domain cgroup')
