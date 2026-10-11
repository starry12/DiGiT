"""Interleave only fresh owned DiGiT VMAs, before any fault or registration.

No process policy, migration, host setting, or GIDS allocation is changed.
Placement receipts are sampled outside the timed training window.
"""
import ctypes as C
import os
from pathlib import Path
from candidates.ukl_real_multi_v9.memory import assert_anonymous

NODES = (0, 1)
PAGE = 4096
MODE = 3                         # MPOL_INTERLEAVE
STATIC = 1 << 15                # Physical node IDs, not cpuset-relative IDs


def ranges(text):
    result = set()
    for part in text.strip().split(','):
        ends = [int(x) for x in part.split('-')]
        if len(ends) not in (1, 2) or min(ends) < 0 or ends[-1] < ends[0]:
            raise ValueError('Invalid node list')
        result.update(range(ends[0], ends[-1]+1))
    return result


def require_nodes():
    allowed = next(line.split(':', 1)[1] for line in
                   Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('Mems_allowed_list:'))
    online = Path('/sys/devices/system/node/online').read_text()
    if not set(NODES) <= ranges(allowed) & ranges(online):
        raise RuntimeError('Both physical NUMA nodes 0 and 1 must be allowed and online')


def api():
    lib = C.CDLL('libnuma.so.1', use_errno=True)
    lib.mbind.argtypes = (C.c_void_p, C.c_ulong, C.c_int,
                         C.POINTER(C.c_ulong), C.c_ulong, C.c_uint)
    lib.mbind.restype = C.c_long
    lib.get_mempolicy.argtypes = (C.POINTER(C.c_int), C.POINTER(C.c_ulong),
                                 C.c_ulong, C.c_void_p, C.c_ulong)
    lib.get_mempolicy.restype = C.c_long
    lib.move_pages.argtypes = (C.c_int, C.c_ulong, C.POINTER(C.c_void_p),
                               C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int)
    lib.move_pages.restype = C.c_long
    return lib


def policies_and_pages(address, size, exact=True):
    """Cover the owned range; full-VMA counters can include merged neighbors."""
    maps = Path('/proc/self/maps').read_text().splitlines()
    numa = {int(p[0], 16): p[1:] for p in
            (line.split() for line in Path('/proc/self/numa_maps').read_text().splitlines())}
    records = []; cursor = address
    for line in maps:
        fields = line.split(); lo, hi = [int(x, 16) for x in fields[0].split('-')]
        if hi <= address or lo >= address+size:
            continue
        if (lo > cursor or (exact and (lo != cursor or hi > address+size))
                or fields[1:2] != ['rw-p'] or fields[3:5] != ['00:00', '0'] or len(fields) != 5):
            raise RuntimeError('NUMA evidence needs exact owned anonymous VMA boundaries')
        tokens = numa.get(lo)
        if not tokens:
            raise RuntimeError('Missing numa_maps evidence')
        values = dict(t.split('=', 1) for t in tokens[1:] if '=' in t)
        pages = {int(k[1:]): int(v) for k, v in values.items() if k.startswith('N') and k[1:].isdigit()}
        records.append(dict(address=lo, bytes=hi-lo, policy=tokens[0], pages=pages,
                            page_kib=int(values.get('kernelpagesize_kB', 4))))
        cursor = min(hi,address+size)
    if cursor != address+size:
        raise RuntimeError('Incomplete NUMA VMA coverage')
    return records


def bind_fresh(arena, label):
    if (arena.closed or arena.address % PAGE or arena.size % PAGE
            or getattr(arena, 'loaded_bytes', 0) or getattr(arena, 'registered', False)):
        raise RuntimeError('NUMA placement requires a fresh unregistered arena')
    require_nodes(); assert_anonymous(arena.address, arena.size)
    # DONTFORK has already separated this mapping from adjacent anonymous VMAs.
    if any(sum(r['pages'].values()) for r in policies_and_pages(arena.address, arena.size)):
        raise RuntimeError('NUMA policy must precede the first page fault')
    lib = api(); mask = C.c_ulong(3)
    if lib.mbind(arena.address, arena.size, MODE | STATIC, C.byref(mask), C.sizeof(mask)*8, 0):
        code = C.get_errno(); raise OSError(code, os.strerror(code))
    # No MPOL_MF_MOVE/STRICT: these are untouched pages; never migrate pinned pages.
    for addr in (arena.address, arena.address+arena.size-PAGE):
        mode = C.c_int(); actual = C.c_ulong()
        if lib.get_mempolicy(C.byref(mode), C.byref(actual), C.sizeof(actual)*8, addr, 2):
            code = C.get_errno(); raise OSError(code, os.strerror(code))
        if mode.value & ~STATIC != MODE or actual.value != 3:
            raise RuntimeError('Effective VMA policy differs from node0/node1 interleave')
    arena.numa_label = label
    return dict(label=label, bytes=arena.size, nodes=list(NODES),
                policy='interleave', before_first_touch=True, migration=False)


def placement(arena):
    records = policies_and_pages(arena.address, arena.size,exact=False)
    # Query physical location for stratified adjacent pairs inside this arena.
    # numa_maps alone cannot attribute per-arena bytes if Linux merged VMAs.
    page_count=arena.size//PAGE
    pairs=min(2048,page_count//2)
    indices=[2*(i*(page_count//2)//pairs)+j for i in range(pairs) for j in (0,1)] if pairs else [0]
    addresses=(C.c_void_p*len(indices))(*(arena.address+i*PAGE for i in indices))
    status=(C.c_int*len(indices))()
    lib=api()
    # nodes=NULL is a location query, never migration or memory allocation.
    if lib.move_pages(0,len(indices),addresses,None,status,0):
        code=C.get_errno();raise OSError(code,os.strerror(code))
    counts = {n: 0 for n in NODES}
    for r in records:
        if r['policy'] != 'interleave=static:0-1' or r['page_kib'] != 4 or set(r['pages']) - set(NODES):
            raise RuntimeError('Unexpected effective NUMA policy, node or page size')
    for node in status:
        if node not in counts:raise RuntimeError('Sample page missing or outside selected NUMA nodes: '+str(node))
        counts[node]+=1
    total = sum(counts.values())
    if total != len(indices) or any(count*5 < total*2 for count in counts.values()):
        raise RuntimeError('NUMA page sample outside 40-60 percent balance')
    return dict(label=arena.numa_label, bytes=arena.size, nodes=list(NODES),
                policy='interleave',sampled_pages=total,node_sample_pages={str(n):v for n,v in counts.items()},
                verification='range_policy_and_stratified_move_pages_query',
                verified=True, before_first_touch=True, migration=False)


def valid_report(report, arm):
    if arm == 'gids':
        return report == {'policy': 'unchanged', 'regions': []}
    try:
        if report['policy'] != 'interleave:0-1' or len(report['regions']) != 3:
            return False
        if {r['label'] for r in report['regions']} != {'graph', 'auxiliary', 'cpu_features'}:
            return False
        for r in report['regions']:
            if not (r['verified'] is True and r['before_first_touch'] is True and r['migration'] is False
                    and r['nodes'] == [0, 1] and r['policy'] == 'interleave'):
                return False
            counts = r['node_sample_pages']
            if set(counts) != {'0', '1'} or type(r['bytes']) is not int or r['bytes'] <= 0:
                return False
            if any(type(v) is not int or v < 0 for v in counts.values()):
                return False
            total=r['sampled_pages']
            if r['verification']!='range_policy_and_stratified_move_pages_query' or type(total)is not int or total!=min(4096,r['bytes']//PAGE//2*2):return False
            if sum(counts.values()) != total or any(v*5 < total*2 for v in counts.values()):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False
