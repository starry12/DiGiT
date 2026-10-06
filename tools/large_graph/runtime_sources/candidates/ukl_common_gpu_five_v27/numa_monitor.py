"""Small proc/sysfs reads; no PTE walk or GPU query in the timed window."""
from pathlib import Path


def snapshot(cgroup=None):
    result = dict(nodes={}, cgroups={})
    try:
        for node in (0, 1):
            p = Path('/sys/devices/system/node/node%d' % node)
            mem = {}
            for line in (p/'meminfo').read_text().splitlines():
                fields = line.split()
                if fields[2] in ('MemTotal:', 'MemFree:', 'Active(anon):', 'Inactive(anon):', 'Unevictable:', 'Mlocked:'):
                    mem[fields[2][:-1]] = int(fields[3])*1024
            vm = dict(line.split() for line in (p/'vmstat').read_text().splitlines())
            result['nodes'][str(node)] = dict(memory=mem, vmstat={k:int(v) for k,v in vm.items()
                if k.startswith(('pgscan', 'pgsteal', 'numa_', 'allocstall', 'compact_'))})
        vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        result['host_reclaim'] = {k:int(v) for k,v in vm.items()
            if k.startswith(('pgscan', 'pgsteal', 'allocstall', 'compact_', 'numa_'))}
        if cgroup:
            root = Path('/sys/fs/cgroup')/cgroup.lstrip('/')
            for name, p in (('service', root), ('init', root/'init'), ('data', root/'data')):
                if not p.exists():
                    continue
                stat = dict(line.split() for line in (p/'memory.stat').read_text().splitlines())
                pressure = {parts[0]: {k:float(v) for k,v in (s.split('=') for s in parts[1:])}
                    for parts in (line.split() for line in (p/'memory.pressure').read_text().splitlines())}
                numa = {parts[0]:dict(s.split('=') for s in parts[1:]) for parts in
                    (line.split() for line in (p/'memory.numa_stat').read_text().splitlines())}
                result['cgroups'][name] = dict(pressure=pressure,numa=numa,
                    reclaim={k:int(v) for k,v in stat.items() if k.startswith(('pgscan','pgsteal','pgfault','pgmajfault'))})
        result['ok'] = True
    except (OSError, ValueError, IndexError) as error:
        # Observation only: the original safety guard remains authoritative.
        result.update(ok=False,error=repr(error))
    return result
