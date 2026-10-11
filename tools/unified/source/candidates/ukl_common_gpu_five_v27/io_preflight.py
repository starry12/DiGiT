"""Qualify the real system-cgroup I/O format using a bounded CPU-only worker."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from . import protocol as P
from .guard import _io_stat
from .tail_probe import expected_tails


def _text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value if isinstance(value, str) else ''


def _write(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(str(temporary), str(path))


def _validate(report, unit, expected_device, evidence):
    if not isinstance(report, dict):
        raise ValueError('Probe report must be a JSON object')
    expected = expected_tails(P.binding())
    payload = sum(v['length'] for v in expected.values())
    requested = 4096*len(expected)
    if (report.get('scope') != 'cpu_cgroup_io_format_probe'
            or type(report.get('read_bytes')) is not int
            or report['read_bytes'] != payload
            or report.get('requested_read_bytes') != requested
            or report.get('reference_bytes') != payload
            or report.get('direct_io') is not True
            or report.get('gpu_called') is not False
            or report.get('raw_ssd_access') is not False
            or report.get('source_device') != expected_device
            or report.get('cgroup') != '/system.slice/' + unit):
        raise ValueError('Probe scope, source, or cgroup does not match')
    arrays = report.get('arrays')
    if not isinstance(arrays, dict) or set(arrays) != set(expected):
        raise ValueError('Probe tail array coverage differs')
    for name, spec in expected.items():
        row = arrays[name]
        if (not isinstance(row, dict) or any(row.get(k) != v for k, v in spec.items())
                or row.get('allocated_bytes') != 4096 or row.get('padding_zero') is not True
                or not isinstance(row.get('source_sha256'), str)
                or len(row['source_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['source_sha256'])
                or row['source_sha256'] != row.get('reference_sha256')):
            raise ValueError('Probe tail bytes, identity or padding mismatch: '+name)
    limits = report.get('limits')
    if not isinstance(limits, dict):
        raise ValueError('Probe effective limits missing')
    memory = int(limits['memory.max'])
    quota, period = map(int, limits['cpu.max'].split())
    if not (0 < memory <= 64 * P.MIB and int(limits['memory.swap.max']) == 0
            and 0 < quota <= period):
        raise ValueError('Probe effective memory, swap, or CPU limit invalid')
    io_limits = {}
    for line in limits['io.max'].splitlines():
        fields = line.split()
        if not fields or fields[0] in io_limits:
            raise ValueError('Probe io.max malformed or duplicated')
        values = {}
        for field in fields[1:]:
            key, value = field.split('=', 1)
            if key in values:
                raise ValueError('Probe io.max field duplicated')
            values[key] = value
        io_limits[fields[0]] = values
    if io_limits.get(expected_device, {}).get('rbps') != str(P.READ_RATE):
        raise ValueError('Probe effective source read limit missing')
    samples = report.get('samples')
    expected_stages = ['before_read', 'after_read'] + ['after_counter_update'] * 3
    if (not isinstance(samples, list) or len(samples) != len(expected_stages)
            or any(not isinstance(row, dict) for row in samples)
            or [row.get('stage') for row in samples] != expected_stages):
        raise ValueError('Probe raw sample lifecycle incomplete')
    evidence['parsed_samples'] = []
    for sample in samples:
        raw = sample.get('raw')
        if not isinstance(raw, str) or len(raw) > 1024 * 1024:
            raise ValueError('Probe raw io.stat missing or oversized')
        evidence['parsed_samples'].append(dict(stage=sample['stage'], io=_io_stat(raw)))
    source = evidence['parsed_samples'][-1]['io'].get(expected_device)
    # EOF may be accounted at sector rather than page granularity. Require
    # every logical byte; requested page-rounded bytes are checked above.
    if not source or source['rbytes'] < payload or source['rios'] < 1:
        raise ValueError('Probe source I/O counters did not observe the direct read')


def run_probe(out, parentunit):
    """Save raw/parsed evidence, returning it on success and raising on failure.

    The caller is the root controller. This function launches only the tiny
    CPU probe; failure must prevent the later CUDA worker from being launched.
    """
    out = Path(out)
    receipt = out / 'cpu_io_stat_probe.json'
    result = dict(passed=False, scope='cpu_cgroup_io_format_probe',
                  parent_unit=parentunit, time=time.time(), stdout='', stderr='')
    started = time.monotonic()
    try:
        device, expected = P.source_device()
        suffix = hashlib.sha256(parentunit.encode()).hexdigest()[:12] + '-' + uuid.uuid4().hex[:12]
        unit = 'digit-ukl-iostat-probe-' + suffix + '.service'
        properties = [
            'Type=exec', 'User=embed', 'WorkingDirectory=' + str(P.ROOT),
            'MemoryAccounting=yes', 'IOAccounting=yes',
            'MemoryMax=' + str(64 * P.MIB), 'MemorySwapMax=0',
            'CPUQuota=100%', 'TasksMax=16', 'RuntimeMaxSec=10',
            'TimeoutStopSec=2', 'KillMode=control-group', 'Restart=no',
            'IOReadBandwidthMax=' + device + ' ' + str(P.READ_RATE),
            'NoNewPrivileges=yes', 'ProtectSystem=strict', 'ProtectHome=read-only',
            'ReadOnlyPaths=/mnt/n0', 'DevicePolicy=closed',
            'InaccessiblePaths=-/dev/libnvm0 -/dev/nvidiactl -/dev/nvidia0 '
            '-/dev/nvidia1 -/dev/nvidia2 -/dev/nvidia3 -/dev/nvidia-uvm '
            '-/dev/nvidia-uvm-tools', 'UMask=0022',
        ]
        command = ['/usr/bin/systemd-run', '--unit=' + unit,
                   '--wait', '--pipe', '--collect', '--quiet']
        command += ['--property=' + value for value in properties]
        command += ['--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=CUDA_VISIBLE_DEVICES=',
                    '/usr/bin/python3', '-B', '-m',
                    'candidates.ukl_common_gpu_five_v27.probe_io_stat']
        result.update(unit=unit, expected_source_device=expected, command=command)
        completed = subprocess.run(command, capture_output=True, text=True, timeout=25,
                                   check=False)
        result.update(returncode=completed.returncode, stdout=_text(completed.stdout),
                      stderr=_text(completed.stderr))
        if completed.returncode != 0:
            raise RuntimeError('CPU probe service exited with status ' + str(completed.returncode))
        if len(result['stdout']) > 1024 * 1024:
            raise ValueError('Probe report exceeds size bound')
        report = json.loads(result['stdout'])
        result['report'] = report
        _validate(report, unit, expected, result)
        result['passed'] = True
    except Exception as failure:
        if isinstance(failure, subprocess.TimeoutExpired):
            result.update(stdout=_text(failure.stdout), stderr=_text(failure.stderr),
                          timed_out=True)
        result['error'] = repr(failure)
        raise RuntimeError('CPU cgroup I/O preflight failed; see ' + str(receipt)) from failure
    finally:
        result['seconds'] = time.monotonic() - started
        _write(receipt, result)
    return result
