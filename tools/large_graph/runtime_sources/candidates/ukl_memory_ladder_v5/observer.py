import json,time
from pathlib import Path

def read(path):
    try:
        return path.read_text()
    except OSError as e:
        return {'error': type(e).__name__}


def counters(text):
    if isinstance(text, dict):
        return text
    return {k.rstrip(':'): int(v) for k, v in (line.split() for line in text.splitlines())}


def process(pid, proc=Path('/proc')):
    p = proc / str(pid)
    stat = read(p / 'stat')
    if isinstance(stat, dict):
        return {'pid': int(pid), **stat}
    # comm may contain spaces and parentheses; starttime disambiguates PID reuse.
    fields = stat[stat.rfind(')') + 2:].split()
    return dict(pid=int(pid), start_ticks=int(fields[19]), state=fields[0],
                comm=read(p/'comm'), io=counters(read(p/'io')),
                cgroup=read(p/'cgroup'))


class Observer:
    def __init__(self, log, proc=Path('/proc'), cg=Path('/sys/fs/cgroup')):
        self.log, self.proc, self.cg = log, proc, cg
        self.unit = None
        self.previous = {}

    def sample(self, memory):
        begin = time.monotonic()
        groups = [self.cg/'user.slice', self.cg/'system.slice']
        groups += list((self.cg/'user.slice').glob('user-*.slice'))
        groups += list((self.cg/'user.slice').glob('user-*.slice/session-*.scope'))
        if self.unit:
            groups.append(self.cg/'system.slice'/self.unit)
        records = []
        for p in groups:
            m = counters(read(p/'memory.stat'))
            records.append(dict(path=str(p), memory={k:v for k,v in m.items()
                if k in ('file_dirty','file_writeback','file','anon','error')},
                io_stat=read(p/'io.stat')))
        processes = []
        for p in self.proc.iterdir():
            if not p.name.isdecimal():
                continue
            item = process(p.name, self.proc)
            key = (item['pid'], item.get('start_ticks'))
            io = item.get('io', {})
            old = self.previous.get(key)
            if 'write_bytes' in io:
                current = io['write_bytes']
                item['write_bytes_delta'] = None if old is None else current-old
                self.previous[key] = current
            # Capture all counters, including permission errors; no argv/env or file contents.
            processes.append(item)
        row = dict(time=time.time(), memory=memory, unit=self.unit,
                   cgroups=records, processes=processes,
                   collection_seconds=time.monotonic()-begin)
        self.log.write(json.dumps(row)+'\n')
        self.log.flush()
        return row


