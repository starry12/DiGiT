"""Read-only host/cgroup observation and fail-closed UKL loading guards.

All sizes are bytes; PSI averages are percentages. This module neither changes
system settings nor stops processes. Callers must stop their own worker on a
negative decision, preserve the observations, and continue bounded exit checks.
"""
import json
import math
import os
import re
import selectors
import subprocess
import time
from pathlib import Path

MIB = 1024 ** 2
GIB = 1024 ** 3
KERNEL_BAD = re.compile(
    r'Xid|NVRM.*(?:error|fail)|ext4.*(?:error|warning)|does not have buffers|'
    r'BUG:|WARNING:|Call Trace:|blocked for more than|I/O error|corrupt', re.I)


def _read(path, limit=1024 * 1024):
    with Path(path).open('r') as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Observation exceeds size bound: ' + str(path))
    return value


def _pairs(text):
    return {row.split()[0]: int(row.split()[1]) for row in text.splitlines()
            if row.strip()}


def _io_stat(text):
    """Parse cgroup v2 I/O records without treating partial counters as zero.

    Linux v5.15 blkcg_print_one_stat omits all base counters when the four
    R/W counters are zero, and can omit the newline after a bare device.
    blk-iocost's ioc_pd_stat and the core debug printer can still emit extras.
    Only those verified complete extras groups establish an extras-only zero
    R/W record. No inference is made about absent discard counters.
    """
    required = ('rbytes', 'wbytes', 'rios', 'wios')
    base = set(required)
    cost_debug = {'cost.wait', 'cost.indebt', 'cost.indelay'}
    delay = {'use_delay', 'delay_nsec'}
    zero_extras = {'cost.usage', 'cost.vrate'} | cost_debug | delay
    result = {}

    def finish(device, fields):
        present = base.intersection(fields)
        if present and present != base:
            raise ValueError('io.stat partial R/W counters for ' + device)
        if not present:
            names = set(fields)
            if names - zero_extras:
                raise ValueError('io.stat unverified counter-free record for ' + device)
            cost = {name for name in names if name.startswith('cost.')}
            if cost and ('cost.usage' not in cost
                         or (cost & cost_debug and cost & cost_debug != cost_debug)):
                raise ValueError('io.stat partial cost fields for ' + device)
            if names & delay and names & delay != delay:
                raise ValueError('io.stat partial delay fields for ' + device)
        result[device] = {key: int(fields[key]) if present else 0 for key in required}

    for line in text.splitlines():
        device, fields = None, {}
        for token in line.split():
            if '=' not in token:
                if not re.fullmatch(r'(?:0|[1-9][0-9]*):(?:0|[1-9][0-9]*)', token):
                    raise ValueError('io.stat malformed device: ' + token)
                major, minor = (int(part) for part in token.split(':'))
                if major >= 2 ** 12 or minor >= 2 ** 20:
                    raise ValueError('io.stat device outside Linux dev_t range: ' + token)
                if device is not None:
                    # Only a bare device can lack its newline in the kernel.
                    if fields:
                        raise ValueError('io.stat missing record newline for ' + device)
                    finish(device, fields)
                if token in result:
                    raise ValueError('io.stat duplicate device: ' + token)
                device, fields = token, {}
                continue
            if device is None:
                raise ValueError('io.stat field without a device: ' + token)
            key, raw = token.split('=', 1)
            if not re.fullmatch(r'[a-z][a-z0-9_.]*', key) or key in fields:
                raise ValueError('io.stat malformed or duplicate field: ' + key)
            integer = key in base | {'dbytes', 'dios'} | (zero_extras - {'cost.vrate'})
            pattern = (r'[0-9]+' if integer else r'[0-9]+\.[0-9]{2}'
                       if key == 'cost.vrate' else r'[0-9]+(?:\.[0-9]+)?')
            if not re.fullmatch(pattern, raw):
                raise ValueError('io.stat invalid nonnegative value for ' + key)
            if integer and int(raw) >= 2 ** 64:
                raise ValueError('io.stat counter outside uint64 range for ' + key)
            fields[key] = raw
        if device is not None:
            finish(device, fields)
    return result


def _psi(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        raw = dict(item.split('=', 1) for item in fields[1:])
        result[fields[0]] = dict(avg10=float(raw['avg10']), total=int(raw['total']))
    for kind in ('some', 'full'):
        if not math.isfinite(result[kind]['avg10']) or result[kind]['avg10'] < 0:
            raise ValueError('Invalid PSI ' + kind)
    return result


def _node_identity(node):
    """Use topology metadata; never read the NVMe uuid fallback attribute."""
    node = node.resolve(strict=True)
    stat = node.stat()
    value = dict(path=str(node), device=stat.st_dev, inode=stat.st_ino)
    if (node / 'diskseq').exists():
        value['diskseq'] = int(_read(node / 'diskseq').strip())
    return value


def _nvme_namespace(node):
    nguid = _read(node / 'nguid').strip().lower()
    if not (re.fullmatch(r'[0-9a-f]{32}', nguid)
            or re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', nguid)):
        raise ValueError('Invalid NVMe NGUID: ' + str(node))
    nguid = nguid.replace('-', '')
    nsid = _read(node / 'nsid').strip()
    if int(nguid, 16) == 0 or not re.fullmatch(r'[0-9]+', nsid) or not 0 < int(nsid) < 2 ** 32 - 1:
        raise ValueError('Missing/reserved NVMe NGUID or NSID: ' + str(node))
    return dict(nguid=nguid, nsid=int(nsid))


def _diskstats(procroot, sysroot):
    devices, excluded = {}, {}
    multipath_heads = set()
    for line in _read(procroot / 'diskstats').splitlines():
        fields = line.split()
        if len(fields) < 14:
            raise ValueError('Malformed diskstats record')
        device, name = ':'.join(fields[:2]), fields[2]
        node = sysroot / 'dev/block' / device
        namespace = None
        if not node.exists():
            # Linux NVMe multipath can expose real per-path counters as 0:0.
            # Verify the hidden namespace via sysfs instead of dropping root IO.
            match = re.fullmatch(r'(nvme\d+)c\d+(n\d+)', name)
            hidden = sysroot / 'class/block' / name
            if device != '0:0' or match is None or not hidden.exists():
                raise ValueError('Cannot resolve diskstats device ' + device)
            head_name = match.group(1)+match.group(2)
            head = sysroot / 'class/block' / head_name
            if _read(hidden/'hidden').strip() != '1' or not head.exists():
                raise ValueError('Unverified NVMe multipath identity: '+name)
            namespace = _nvme_namespace(hidden)
            if namespace != _nvme_namespace(head):
                raise ValueError('Unverified NVMe multipath namespace: '+name)
            namespace['head'] = _node_identity(head)
            multipath_heads.add(head_name)
            device, node = '0:0:'+name, hidden
        if (node / 'partition').exists():
            excluded[device] = 'partition'
            continue
        slaves = node / 'slaves'
        if slaves.exists() and any(slaves.iterdir()):
            excluded[device] = 'stacked_device'
            continue
        # Linux diskstats sectors are always 512 bytes, regardless of sector size.
        identity = _node_identity(node)
        if namespace is not None:
            identity['namespace'] = namespace
        pending = _read(node / 'inflight').split()
        if len(pending) != 2 or any(not re.fullmatch(r'[0-9]+', item) for item in pending):
            raise ValueError('Invalid read/write inflight observation: ' + name)
        devices[device] = dict(name=name, rbytes=int(fields[5]) * 512,
                               wbytes=int(fields[9]) * 512,
                               rios=int(fields[3]), wios=int(fields[7]),
                               inflight=int(fields[11]), io_ms=int(fields[12]),
                               weighted_io_ms=int(fields[13]), identity=identity,
                               read_inflight=int(pending[0]), write_inflight=int(pending[1]))
    for device, values in list(devices.items()):
        if values['name'] in multipath_heads:
            excluded[device] = 'multipath_head_counted_by_verified_paths'
            del devices[device]
    if not devices:
        raise ValueError('No resolved diskstats devices')
    totals = {key: sum(item[key] for item in devices.values())
              for key in ('rbytes', 'wbytes', 'rios', 'wios')}
    return dict(devices=devices, totals=totals, excluded=excluded)


def collect(unit_cgroup=None, procroot='/proc', cgroot='/sys/fs/cgroup',
            sysroot='/sys'):
    """Return a JSON observation; missing/malformed required data goes in errors.

    unit_cgroup is systemd ControlGroup (e.g. /system.slice/example.service),
    not a PID. None omits worker metrics for admission/post-exit observation.
    """
    procroot, cgroot, sysroot = Path(procroot), Path(cgroot), Path(sysroot)
    value = dict(time=time.time(), monotonic=time.monotonic(), errors=[], error_details=[])

    def capture(name, function):
        try:
            value[name] = function()
        except (OSError, ValueError, KeyError, IndexError) as error:
            message = name + ': ' + str(error)
            value['errors'].append(message)
            value['error_details'].append(dict(component=name,
                exception=type(error).__name__, errno=getattr(error, 'errno', None),
                filename=getattr(error, 'filename', None), message=message))

    def memory():
        raw = _pairs(_read(procroot / 'meminfo'))
        return dict(available=raw['MemAvailable:'] * 1024,
                    dirty=raw['Dirty:'] * 1024, writeback=raw['Writeback:'] * 1024)

    def vmstat():
        raw = _pairs(_read(procroot / 'vmstat'))
        return {key: raw[key] for key in ('nr_written', 'pgpgout')}

    def cgroup():
        path = (cgroot / str(unit_cgroup).lstrip('/')).resolve()
        path.relative_to(cgroot.resolve())
        stats, events = (_pairs(_read(path / name))
                         for name in ('memory.stat', 'memory.events'))
        result = dict(path=str(unit_cgroup), current=int(_read(path / 'memory.current')),
                      events={key: events[key] for key in ('high', 'max', 'oom', 'oom_kill')})
        result.update({key: stats[key] for key in ('file_dirty', 'file_writeback', 'file', 'anon', 'shmem')})
        # Retain exact evidence even if parsing raises and capture rejects cgroup.
        raw_io = _read(path / 'io.stat')
        value['cgroup_io_stat_raw'] = raw_io
        result['io'] = _io_stat(raw_io)
        return result

    capture('memory', memory)
    capture('psi', lambda: {name: _psi(_read(procroot / 'pressure' / name))
                            for name in ('io', 'memory')})
    capture('boot_id', lambda: _read(procroot / 'sys/kernel/random/boot_id').strip())
    capture('vmstat', vmstat)
    capture('diskstats', lambda: _diskstats(procroot, sysroot))
    if unit_cgroup is not None:
        capture('cgroup', cgroup)
    value['sample_seconds'] = time.monotonic() - value['monotonic']
    return value


def _validate(sample, require_cgroup):
    reasons = list(sample.get('errors', []))
    try:
        for key in ('time', 'monotonic'):
            if not math.isfinite(sample[key]):
                raise ValueError('Nonfinite observation clock')
        if not sample['boot_id']:
            raise ValueError('Missing boot identity')
        if sample.get('sample_seconds', 0) > 5.5:
            raise ValueError('Observation took over 5.5 seconds')
        for key in ('available', 'dirty', 'writeback'):
            if sample['memory'][key] < 0:
                raise ValueError('Negative memory observation')
        for resource in ('io', 'memory'):
            for kind in ('some', 'full'):
                if not math.isfinite(sample['psi'][resource][kind]['avg10']):
                    raise ValueError('Nonfinite PSI')
                if sample['psi'][resource][kind]['avg10'] < 0:
                    raise ValueError('Negative PSI')
                if sample['psi'][resource][kind]['total'] < 0:
                    raise ValueError('Negative PSI total')
        for key in ('nr_written', 'pgpgout'):
            if sample['vmstat'][key] < 0:
                raise ValueError('Negative writeback counter')
        for key in ('rbytes', 'wbytes', 'rios', 'wios'):
            if sample['diskstats']['totals'][key] < 0:
                raise ValueError('Negative diskstats counter')
        if not sample['diskstats']['devices']:
            raise ValueError('Missing diskstats devices')
        for device in sample['diskstats']['devices'].values():
            if not device['name'] or not isinstance(device['identity'], dict) or not device['identity']:
                raise ValueError('Missing device identity')
            for key in ('rbytes', 'wbytes', 'rios', 'wios', 'inflight', 'io_ms', 'weighted_io_ms',
                        'read_inflight', 'write_inflight'):
                if device[key] < 0:
                    raise ValueError('Negative device counter')
        if require_cgroup:
            cg = sample['cgroup']
            if not cg['path']:
                raise ValueError('Missing cgroup path')
            for key in ('current', 'file_dirty', 'file_writeback', 'file', 'anon'):
                if cg[key] < 0:
                    raise ValueError('Negative cgroup observation')
            for key in ('high', 'max', 'oom', 'oom_kill'):
                if cg['events'][key] < 0:
                    raise ValueError('Negative cgroup event')
            if not isinstance(cg['io'], dict):
                raise ValueError('Missing cgroup I/O observation')
    except (KeyError, ValueError, TypeError) as error:
        reasons.append('monitor_unavailable: ' + str(error))
    return reasons


def _avg(sample, resource, kind):
    return sample['psi'][resource][kind]['avg10']


class Guard:
    """Runtime guard. All threshold durations use monotonic elapsed time.

    A global Dirty increase alone is recorded, not rejected. A 30-second
    writeback stall needs Dirty >=256 MiB, pending device writes without
    completed writes on that same device, and I/O PSI
    some >=10% or full >=5%. Independently, 30 seconds of I/O PSI some >=30%
    or full >=20%, or memory PSI some >=10% or full >=5%, stops the worker.
    This is a stop request, not a guarantee that a blocked kernel can stop.
    """
    def __init__(self, require_cgroup=True, max_gap=5.5, duration=30.0, file_cache_limit=128*MIB):
        self.file_cache_limit=file_cache_limit
        self.require_cgroup, self.max_gap, self.duration = require_cgroup, max_gap, duration
        self.before, self.previous, self.stall_since, self.pressure_since = None, None, None, None
        self.failed = []
        self.device_highwater, self.device_regressions, self.device_stalls = {}, {}, {}
        self.vm_highwater = {}

    def _progress(self, sample, reasons, observations):
        """Keep approximate VM counters separate from device completions.

        One diskstats regression requests a fresh sample, preserving all stall
        timers and the last trusted high water. Two consecutive regressions
        fail closed. VM regressions are diagnostic only and cannot clear timers.
        """
        device_progress, vm = {}, {}
        for key in ('nr_written', 'pgpgout'):
            value = sample['vmstat'][key]
            high = self.vm_highwater.get(key, value)
            delta = 0 if self.previous is None else value - self.previous['vmstat'][key]
            vm[key] = dict(delta=delta, high_water=max(high, value),
                           high_water_increment=max(0, value - high), jitter=delta < 0)
            if delta < 0:
                observations.append('vmstat_counter_jitter:' + key)
            self.vm_highwater[key] = max(high, value)
        counters = ('rbytes', 'wbytes', 'rios', 'wios', 'io_ms', 'weighted_io_ms')
        for key, current in sample['diskstats']['devices'].items():
            high = self.device_highwater.get(key)
            if high is None:
                self.device_highwater[key] = {field: current[field] for field in counters}
                device_progress[key] = dict(trusted=True, baseline=True, completed=False,
                                             write_progress=False, busy=current['inflight'] > 0,
                                             pending_writes=current['write_inflight'] > 0)
                continue
            regressed = [field for field in counters if current[field] < high[field]]
            if regressed:
                count = self.device_regressions.get(key, 0) + 1
                self.device_regressions[key] = count
                observations.append('diskstats_counter_retry:' + key)
                if count >= 2:
                    reasons.append('diskstats_counter_persistent_regression:' + key)
                device_progress[key] = dict(trusted=False, regressed=regressed,
                                             consecutive=count, completed=False,
                                             write_progress=False, busy=current['inflight'] > 0,
                                             pending_writes=current['write_inflight'] > 0)
                continue
            delta = {field: current[field] - high[field] for field in counters}
            if self.device_regressions.pop(key, 0):
                observations.append('diskstats_counter_recovered:' + key)
            self.device_highwater[key] = {field: current[field] for field in counters}
            device_progress[key] = dict(trusted=True, baseline=False, delta=delta,
                                         completed=any(delta[field] > 0 for field in counters[:4]),
                                         write_progress=delta['wbytes'] > 0 or delta['wios'] > 0,
                                         pending_writes=current['write_inflight'] > 0,
                                         busy=current['inflight'] > 0 or delta['io_ms'] > 0
                                         or delta['weighted_io_ms'] > 0)
        return device_progress, vm

    def check(self, sample):
        unavailable = _validate(sample, self.require_cgroup)
        reasons = list(self.failed) + unavailable
        observations = []
        if unavailable:
            self.failed = list(dict.fromkeys(reasons))
            return dict(ok=False, reasons=self.failed, observations=observations)
        now, mem = sample['monotonic'], sample['memory']
        previous = self.previous
        initial = self.before is None
        if initial:
            self.before = sample
        if sample['boot_id'] != self.before['boot_id']:
            reasons.append('boot_changed')
        if previous:
            gap = now - previous['monotonic']
            if gap <= 0 or gap > self.max_gap:
                reasons.append('monitor_gap')
            if set(sample['diskstats']['devices']) != set(previous['diskstats']['devices']):
                reasons.append('diskstats_device_set_changed')
            if sample['diskstats']['excluded'] != previous['diskstats']['excluded']:
                reasons.append('diskstats_topology_changed')
            for device in set(sample['diskstats']['devices']) & set(previous['diskstats']['devices']):
                current, old = sample['diskstats']['devices'][device], previous['diskstats']['devices'][device]
                if current['name'] != old['name'] or current['identity'] != old['identity']:
                    reasons.append('diskstats_device_identity_changed:' + device)
        if mem['available'] < 64 * GIB:
            reasons.append('host_available_below_64gib')
        if mem['dirty'] >= 32 * GIB:
            reasons.append('global_dirty_32gib')
        elif mem['dirty'] >= 2 * GIB:
            observations.append('global_dirty_above_2gib_attribution_warning')
        if mem['writeback'] >= 256 * MIB:
            reasons.append('global_writeback_256mib')
        if mem['dirty'] - self.before['memory']['dirty'] > 256 * MIB:
            observations.append('external_dirty_rise_requires_attribution')
        if self.require_cgroup:
            cg, first = sample['cgroup'], self.before['cgroup']
            if cg['path'] != first['path']:
                reasons.append('worker_cgroup_changed')
            if cg['file_dirty'] > 64 * MIB:
                reasons.append('own_dirty_above_64mib')
            if cg['file_writeback'] > 64 * MIB:
                reasons.append('own_writeback_above_64mib')
            if cg['file'] > self.file_cache_limit:
                reasons.append('own_file_cache_above_128mib' if self.file_cache_limit == 128*MIB else 'aggregate_file_cache_limit')
            for event in ('max', 'oom', 'oom_kill'):
                # A newly-created worker cgroup must begin with zero hard events;
                # a hit before our first poll must not become an accepted baseline.
                if (initial and cg['events'][event] > 0) or cg['events'][event] > first['events'][event]:
                    reasons.append('cgroup_event_' + event)
                if previous and cg['events'][event] < previous['cgroup']['events'][event]:
                    reasons.append('cgroup_event_counter_reset')
            if cg['events']['high'] > first['events']['high']:
                observations.append('cgroup_memory_high_increment')
        high_pressure = (_avg(sample, 'io', 'some') >= 30 or _avg(sample, 'io', 'full') >= 20
                         or _avg(sample, 'memory', 'some') >= 10 or _avg(sample, 'memory', 'full') >= 5)
        self.pressure_since = (now if self.pressure_since is None else self.pressure_since) if high_pressure else None
        if self.pressure_since is not None and now - self.pressure_since >= self.duration:
            reasons.append('sustained_pressure_30s')
        device_progress, vm = self._progress(sample, reasons, observations)
        writeback_pressure = (mem['dirty'] >= 256 * MIB
                              and (_avg(sample, 'io', 'some') >= 10 or _avg(sample, 'io', 'full') >= 5))
        for device, progress in device_progress.items():
            since = self.device_stalls.get(device)
            if not writeback_pressure:
                self.device_stalls.pop(device, None)
            elif not progress['trusted']:
                # Unknown counter progress is never evidence of recovery.
                if since is None and progress['pending_writes'] and previous is not None:
                    self.device_stalls[device] = now
            elif previous is not None and not progress['write_progress'] and progress['pending_writes']:
                self.device_stalls[device] = now if since is None else since
            else:
                # Reads cannot establish a writeback stall, nor clear one while
                # writes remain pending. No pending write is not a write stall.
                self.device_stalls.pop(device, None)
        self.stall_since = min(self.device_stalls.values()) if self.device_stalls else None
        if self.stall_since is not None and now - self.stall_since >= self.duration:
            reasons.append('writeback_stall_30s')
        self.previous = sample
        self.failed = list(dict.fromkeys(reasons))
        return dict(ok=not reasons, reasons=self.failed, observations=observations,
                    device_progress=device_progress, vmstat_observation=vm,
                    stall_devices={key: now - value for key, value in self.device_stalls.items()},
                    pressure_seconds=0 if self.pressure_since is None else now - self.pressure_since,
                    stall_seconds=0 if self.stall_since is None else now - self.stall_since)


class AdmissionWindow:
    """Require 30 continuous quiet seconds; never run or queue a worker itself.

    source_devices are physical major:minor diskstats keys. An empty tuple
    conservatively checks all resolved nonpartition/nonstacked block devices.
    """
    def __init__(self, host_min, source_devices=(), seconds=30.0, max_gap=5.5):
        self.host_min, self.source_devices = host_min, tuple(source_devices)
        self.seconds, self.max_gap = seconds, max_gap
        self.previous, self.quiet_since, self.boot_id = None, None, None
        self.guard = Guard(require_cgroup=False, max_gap=max_gap)

    def check(self, sample):
        reasons = _validate(sample, False)
        fatal = bool(reasons)
        if reasons:
            self.quiet_since = None
            return dict(ok=False, ready=False, reasons=reasons, stable_seconds=0)
        guard = self.guard.check(sample)
        if not guard['ok']:
            self.quiet_since = None
            return dict(ok=False, ready=False, reasons=guard['reasons'], stable_seconds=0,
                        guard=guard)
        if any(not progress['trusted'] for progress in guard['device_progress'].values()):
            reasons.append('diskstats_counter_resample')
        now, mem, rate = sample['monotonic'], sample['memory'], None
        if self.boot_id is None:
            self.boot_id = sample['boot_id']
        if sample['boot_id'] != self.boot_id:
            reasons.append('boot_changed')
            fatal = True
        if mem['available'] < self.host_min:
            reasons.append('host_memory_budget')
        if mem['dirty'] >= 128 * MIB or mem['writeback'] >= 64 * MIB:
            reasons.append('host_not_quiet_dirty_writeback')
        if any(_avg(sample, resource, kind) >= 1 for resource in ('io', 'memory') for kind in ('some', 'full')):
            reasons.append('host_not_quiet_psi')
        if self.previous is None:
            reasons.append('need_diskstats_interval')
        else:
            elapsed = now - self.previous['monotonic']
            if elapsed <= 0 or elapsed > self.max_gap:
                reasons.append('monitor_gap')
                fatal = True
            else:
                devices = self.source_devices or tuple(sample['diskstats']['devices'])
                try:
                    delta = 0
                    for device in devices:
                        current = sample['diskstats']['devices'][device]
                        previous = self.previous['diskstats']['devices'][device]
                        if not guard['device_progress'][device]['trusted']:
                            reasons.append('source_device_counter_resample')
                            continue
                        for key in ('rbytes', 'wbytes'):
                            change = current[key] - previous[key]
                            delta += change
                    rate = delta / elapsed
                    if rate > 32 * MIB:
                        reasons.append('source_device_busy')
                except (KeyError, ValueError) as error:
                    reasons.append('source_device_monitor_unavailable: ' + str(error))
                    fatal = True
        self.quiet_since = None if reasons else (now if self.quiet_since is None else self.quiet_since)
        stable = 0 if self.quiet_since is None else now - self.quiet_since
        self.previous = sample
        return dict(ok=not fatal, ready=not reasons and stable >= self.seconds,
                    reasons=reasons, stable_seconds=stable, source_io_bytes_per_second=rate,
                    guard=guard)


def _bounded_command(argv, timeout, max_bytes):
    """Read both journal pipes within one strict shared byte/time budget."""
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selector, output = selectors.DefaultSelector(), {'stdout': bytearray(), 'stderr': bytearray()}
    deadline, total = time.monotonic() + timeout, 0
    try:
        for label, stream in (('stdout', process.stdout), ('stderr', process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Kernel journal query timeout')
            for key, _ in selector.select(remaining):
                chunk = os.read(key.fileobj.fileno(), min(8192, max_bytes - total + 1))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError('Kernel journal output overflow')
                output[key.data].extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Kernel journal query timeout')
        if process.wait(timeout=remaining) != 0 or output['stderr'].strip():
            raise RuntimeError('Kernel journal query failed: ' + output['stderr'].decode('utf-8', 'replace')[:512])
        return output['stdout'].decode('utf-8', 'strict')
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        process.stdout.close()
        process.stderr.close()


class KernelCursor:
    """Capture a baseline, then consume every new kernel record without -n.

    Call poll at least every 2 seconds during loading, registration and release.
    Overflow/timeouts/parse errors permanently prevent acceptance, but polling
    continues from the last completely validated cursor to preserve evidence.
    Establish ignores old messages; poll fails on every new priority <=4 or
    known GPU/filesystem/kernel error. No alert is considered recoverable here.
    """
    def __init__(self, runner=None, procroot='/proc', timeout=2.0, max_bytes=1024 * 1024):
        self.runner = runner or _bounded_command
        self.procroot, self.timeout, self.max_bytes = Path(procroot), timeout, max_bytes
        self.cursor, self.boot_id, self.failed = None, None, None
        self.first_failure = None
        self.boot_invalid = False
        self.seen_cursors = set()

    def _failure(self, error, source):
        if self.failed is None:
            self.failed = error
            self.first_failure = dict(error=error, source=source,
                                      cursor=self.cursor, boot_id=self.boot_id)

    def _result(self, query_ok, records, alerts, current_error=None):
        return dict(ok=self.failed is None, query_ok=query_ok, error=self.failed,
                    current_error=current_error, first_failure=self.first_failure,
                    records=records, alerts=alerts, cursor=self.cursor,
                    boot_id=self.boot_id, boot_invalid=self.boot_invalid)

    def _query(self, initial):
        source = 'boot_identity'
        try:
            if self.boot_invalid:
                raise RuntimeError('Kernel monitor boot identity invalidated; old cursor cannot be reused')
            boot = _read(self.procroot / 'sys/kernel/random/boot_id').strip()
            if not boot or (self.boot_id is not None and boot != self.boot_id):
                self.boot_invalid = True
                raise RuntimeError('Kernel monitor boot identity changed/missing')
            # Bind this instance to one boot even if its first journal query fails.
            self.boot_id = boot
            source = 'cursor_state'
            argv = ['/usr/bin/journalctl', '-k', '-b', '-o', 'json', '--show-cursor', '--no-pager', '--quiet']
            if initial:
                argv += ['-n', '1']
            elif self.cursor is None:
                raise RuntimeError('Kernel monitor has no baseline cursor')
            else:
                argv += ['--after-cursor=' + self.cursor]
            source = 'journal_query'
            text = self.runner(argv, self.timeout, self.max_bytes)
            if len(text.encode('utf-8')) > self.max_bytes:
                raise RuntimeError('Kernel journal output overflow')
            source = 'journal_parse'
            records, cursor, batch_cursors = [], None, set()
            for line in text.splitlines():
                if not line.strip() or line == '-- No entries --':
                    continue
                if line.startswith('-- cursor: '):
                    if cursor is not None:
                        raise ValueError('Duplicate journal cursor')
                    cursor = line[len('-- cursor: '):].strip()
                    if not cursor:
                        raise ValueError('Empty journal cursor')
                else:
                    if cursor is not None:
                        raise ValueError('Kernel records follow journal cursor trailer')
                    record = json.loads(line)
                    if (not isinstance(record, dict)
                            or not isinstance(record.get('__CURSOR'), str)
                            or not record['__CURSOR'].strip()):
                        raise ValueError('Kernel record missing cursor')
                    record_cursor = record['__CURSOR']
                    if record_cursor in batch_cursors or record_cursor in self.seen_cursors:
                        raise ValueError('Kernel journal repeated an already observed cursor')
                    if record.get('_BOOT_ID') != boot.replace('-', ''):
                        raise ValueError('Kernel record boot identity mismatch')
                    if not isinstance(record.get('MESSAGE'), str):
                        raise ValueError('Kernel record MESSAGE is not text')
                    priority = int(record['PRIORITY'])
                    if priority < 0 or priority > 7:
                        raise ValueError('Invalid kernel priority')
                    records.append(record)
                    batch_cursors.add(record_cursor)
            if records and (not cursor or cursor != records[-1]['__CURSOR']):
                raise ValueError('Kernel journal cursor not aligned with records')
            if initial and (not cursor or len(records) != 1):
                raise ValueError('Cannot establish kernel journal cursor')
            if not records and cursor is not None and cursor != self.cursor:
                raise ValueError('Kernel cursor advanced without records')
            # Commit only after the entire bounded query has been validated.
            # A query/parse failure will retry from the previous valid cursor.
            self.cursor = cursor or self.cursor
            self.seen_cursors.update(batch_cursors)
            alerts = [] if initial else [record for record in records
                                        if int(record['PRIORITY']) <= 4 or KERNEL_BAD.search(record['MESSAGE'])]
            if alerts:
                self._failure('New kernel alert', 'kernel_alert')
            return self._result(True, records, alerts, 'New kernel alert' if alerts else None)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            self._failure(str(error), source)
            return self._result(False, [], [], str(error))

    def establish(self):
        if self.cursor is not None:
            error = 'Kernel cursor already established'
            self._failure(error, 'cursor_state')
            return self._result(False, [], [], error)
        return self._query(True)

    def poll(self):
        return self._query(False)
