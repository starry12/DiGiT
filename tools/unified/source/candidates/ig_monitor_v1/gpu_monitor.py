"""Persistent, bounded NVML monitoring; this module never imports CUDA or torch."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time

BACKEND = 'persistent_nvml_ctypes_v1'
MEMORY_LIMIT = 64 * 2**20


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def memory(pid):
    fields = {}
    for line in Path('/proc/%d/status' % pid).read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value, *_ = line.split()
            fields[key.rstrip(':')] = int(value) * 1024
    if not all(k in fields for k in ('VmRSS', 'VmHWM')):
        raise RuntimeError('Process memory unavailable: %d' % pid)
    return fields


class NvmlBackend:
    class Memory(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint), ('total', ctypes.c_ulonglong),
                    ('reserved', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                    ('used', ctypes.c_ulonglong)]

    class Utilization(ctypes.Structure):
        _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

    def __init__(self, gpu):
        self.gpu = str(gpu)
        if not self.gpu.isdecimal():
            raise ValueError('One physical numeric GPU index is required')
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.lib.nvmlErrorString.argtypes = [ctypes.c_int]
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self.function('nvmlInit_v2', [])()
        self.handle = ctypes.c_void_p()
        self.function('nvmlDeviceGetHandleByIndex_v2', [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])(
            int(self.gpu), ctypes.byref(self.handle))
        self.uuid = self.string('nvmlDeviceGetUUID', self.handle)
        if not self.uuid.startswith('GPU-'):
            raise RuntimeError('Invalid physical GPU UUID')
        self.driver_version = self.string('nvmlSystemGetDriverVersion')
        self.nvml_version = self.string('nvmlSystemGetNVMLVersion')
        self.get_memory = self.function('nvmlDeviceGetMemoryInfo_v2', [ctypes.c_void_p, ctypes.POINTER(self.Memory)])
        self.get_utilization = self.function('nvmlDeviceGetUtilizationRates', [ctypes.c_void_p, ctypes.POINTER(self.Utilization)])

    def function(self, name, args):
        function = getattr(self.lib, name)
        function.argtypes = args
        function.restype = ctypes.c_int
        def checked(*values):
            code = function(*values)
            if code:
                message = self.lib.nvmlErrorString(code).decode('utf-8', 'replace')
                raise RuntimeError('%s failed (%d): %s' % (name, code, message))
        return checked

    def string(self, name, handle=None):
        buffer = ctypes.create_string_buffer(128)
        args = [ctypes.c_char_p, ctypes.c_uint]
        if handle is None:
            self.function(name, args)(buffer, len(buffer))
        else:
            self.function(name, [ctypes.c_void_p] + args)(handle, buffer, len(buffer))
        return buffer.value.decode('utf-8', 'strict')

    def sample(self):
        mem, util = self.Memory(), self.Utilization()
        mem.version = ctypes.sizeof(self.Memory) | (2 << 24)
        self.get_memory(self.handle, ctypes.byref(mem))
        self.get_utilization(self.handle, ctypes.byref(util))
        return dict(backend=BACKEND, physical_gpu_index=int(self.gpu),
                    physical_gpu_uuid=self.uuid, nvml_version=self.nvml_version,
                    driver_version=self.driver_version, device_used_bytes=int(mem.used),
                    device_total_bytes=int(mem.total), device_reserved_bytes=int(mem.reserved),
                    device_free_bytes=int(mem.free), memory_api='nvmlDeviceGetMemoryInfo_v2',
                    memory_used_definition='NVML v2 used bytes excludes driver/firmware reserved memory',
                    utilization_percent=int(util.gpu))


def sampler_main(gpu):
    # NVML is loaded only in this fresh exec process. The supervisor owns the deadline.
    backend = None
    for line in sys.stdin:
        request = json.loads(line)
        started = time.time()
        try:
            if backend is None:
                backend = NvmlBackend(gpu)
            result = dict(backend.sample(), query_started_unix=started,
                          query_finished_unix=time.time(), sampler_pid=os.getpid(),
                          sampler_parent_pid=os.getppid(), sequence=request['sequence'], ok=True)
        except Exception as exc:
            result = dict(ok=False, sequence=request['sequence'], sampler_pid=os.getpid(),
                          error=type(exc).__name__ + ': ' + str(exc), error_kind='nvml_query_error')
        print(json.dumps(result), flush=True)
        if not result['ok']:
            return 2
    return 0


class StopRequested(Exception):
    pass


class QueryFailure(RuntimeError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def read_response(child, selector, stop, deadline, buffer):
    while True:
        if stop.is_set():
            raise StopRequested()
        if time.monotonic() >= deadline:
            raise QueryFailure('query_timeout', 'Persistent NVML query exceeded its configured deadline')
        if b'\n' in buffer:
            line, remainder = buffer.split(b'\n', 1)
            if remainder:
                raise QueryFailure('protocol_error', 'Unexpected multiple sampler messages')
            try:
                return json.loads(line.decode('utf-8'))
            except (ValueError, UnicodeError) as exc:
                raise QueryFailure('protocol_error', str(exc))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QueryFailure('query_timeout', 'Persistent NVML query exceeded its configured deadline')
        if selector.select(min(remaining, .05)):
            chunk = os.read(child.stdout.fileno(), 65536)
            if not chunk:
                raise QueryFailure('sampler_disconnected', 'NVML sampler closed its response pipe')
            buffer += chunk
            if len(buffer) > 65536:
                raise QueryFailure('protocol_error', 'Oversized sampler message')


def stop_sampler(child):
    if child is None:
        return None, 'not_started'
    prior = child.poll()
    if prior is not None:
        return prior, 'already_exited'
    child.terminate()
    try:
        return child.wait(timeout=2), 'terminated'
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait(timeout=2), 'killed'


def run_monitor(output, gpu, period=.5, timeout=5., sampler_command=None, expected_backend=BACKEND):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not str(gpu).isdecimal() or period <= 0 or timeout <= 0:
        raise ValueError('One physical GPU index and positive timings are required')
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    controller_pid = os.getppid()
    state = dict(schema='digit-persistent-nvml-monitor-v1', mode='external_small_process',
                 backend=expected_backend, backend_version=1, pid=os.getpid(), parent_pid=controller_pid,
                 gpu=str(gpu), physical_gpu_index=int(gpu), physical_gpu_uuid=None,
                 started_unix=time.time(), samples=0, errors=[], peak_device_used_bytes=0,
                 peak_rss_bytes=0, supervisor_peak_rss_bytes=0, sampler_peak_rss_bytes=0,
                 period_after_query_seconds=period, query_timeout_seconds=timeout,
                 host_limit_bytes=MEMORY_LIMIT, sampler_pid=None, complete=False, passed=False)
    child = None
    sequence = 0
    first_identity = None
    supervisor_memory = memory(os.getpid())
    sampler_memory = dict(VmRSS=0, VmHWM=0)
    try:
        with (output/'samples.jsonl').open('x', buffering=1) as stream, (output/'sampler.log').open('x') as log:
            def error(kind, message):
                record = dict(time_unix=time.time(), error=message, error_kind=kind,
                              sequence=sequence, monitor_pid=os.getpid(),
                              monitor_parent_pid=controller_pid, sampler_pid=state['sampler_pid'])
                state['errors'].append(record)
                stream.write(json.dumps(record) + '\n')
            try:
                command = sampler_command or [sys.executable, '-u', str(Path(__file__).resolve()),
                                               '--sampler', '--gpu', str(gpu)]
                child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=log, bufsize=0, close_fds=True)
                state['sampler_pid'] = child.pid
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout, selectors.EVENT_READ)
                    while not stop.is_set():
                        sequence += 1
                        began = time.time()
                        deadline = time.monotonic() + timeout
                        try:
                            child.stdin.write((json.dumps(dict(sequence=sequence))+'\n').encode())
                            response = read_response(child, selector, stop, deadline, b'')
                            if response.get('sequence') != sequence or response.get('sampler_pid') != child.pid:
                                raise QueryFailure('identity_error', 'Wrong sampler sequence or PID')
                            if not response.get('ok'):
                                raise QueryFailure(response.get('error_kind', 'nvml_query_error'), response.get('error', 'Sampler query failed'))
                            if response.get('sampler_parent_pid') != os.getpid():
                                raise QueryFailure('identity_error', 'Wrong sampler ownership')
                            identity = (response.get('backend'), response.get('physical_gpu_index'), response.get('physical_gpu_uuid'))
                            if identity[0] != expected_backend or identity[1] != int(gpu) or not isinstance(identity[2], str) or not identity[2].startswith('GPU-'):
                                raise QueryFailure('identity_error', 'Wrong NVML backend or GPU identity')
                            if first_identity is not None and identity != first_identity:
                                raise QueryFailure('identity_error', 'Physical GPU identity changed during monitoring')
                            if first_identity is None:
                                first_identity = identity
                                state['physical_gpu_uuid'] = identity[2]
                                state['nvml_version'] = response.get('nvml_version')
                                state['driver_version'] = response.get('driver_version')
                            if not (isinstance(response.get('device_used_bytes'), int) and response['device_used_bytes'] >= 0 and isinstance(response.get('utilization_percent'), int) and 0 <= response['utilization_percent'] <= 100):
                                raise QueryFailure('invalid_sample', 'Invalid memory/utilization reading')
                            ended = time.time()
                            if not began <= response['query_started_unix'] <= response['query_finished_unix'] <= ended:
                                raise QueryFailure('invalid_sample', 'Invalid query timestamps')
                            supervisor_memory, sampler_memory = memory(os.getpid()), memory(child.pid)
                            combined_rss = supervisor_memory['VmRSS'] + sampler_memory['VmRSS']
                            combined_peak = supervisor_memory['VmHWM'] + sampler_memory['VmHWM']
                            state['supervisor_peak_rss_bytes'] = max(state['supervisor_peak_rss_bytes'], supervisor_memory['VmHWM'])
                            state['sampler_peak_rss_bytes'] = max(state['sampler_peak_rss_bytes'], sampler_memory['VmHWM'])
                            state['peak_rss_bytes'] = max(state['peak_rss_bytes'], combined_peak)
                            if state['peak_rss_bytes'] > MEMORY_LIMIT:
                                raise QueryFailure('memory_budget', 'Combined monitor processes exceeded 64 MiB')
                            record = dict(response)
                            record.pop('ok')
                            record.update(time_unix=ended, monitor_pid=os.getpid(), monitor_parent_pid=controller_pid,
                                          monitor_rss_bytes=combined_rss, supervisor_rss_bytes=supervisor_memory['VmRSS'],
                                          sampler_rss_bytes=sampler_memory['VmRSS'], supervisor_peak_rss_bytes=supervisor_memory['VmHWM'],
                                          sampler_peak_rss_bytes=sampler_memory['VmHWM'])
                            stream.write(json.dumps(record) + '\n')
                            state['samples'] += 1
                            state['peak_device_used_bytes'] = max(state['peak_device_used_bytes'], record['device_used_bytes'])
                            if state['samples'] == 1:
                                write(output/'ready.json', dict(pid=os.getpid(), parent_pid=controller_pid, passed=True,
                                      sampler_pid=child.pid, backend=expected_backend, physical_gpu_index=int(gpu),
                                      physical_gpu_uuid=state['physical_gpu_uuid'], first_sample=record))
                        except StopRequested:
                            state['stop_during_query'] = True
                            break
                        except Exception as exc:
                            error(getattr(exc, 'kind', 'supervisor_error'), type(exc).__name__ + ': ' + str(exc))
                            break
                        if os.getppid() != controller_pid:
                            error('controller_disconnected', 'Controller process exited without stopping monitor')
                            break
                        stop.wait(period)
            except Exception as exc:
                error(getattr(exc, 'kind', 'supervisor_error'), type(exc).__name__ + ': ' + str(exc))
            finally:
                # Take the last measurable process high-water values before bounded teardown.
                for pid, key in ((os.getpid(), 'supervisor_peak_rss_bytes'), (state['sampler_pid'], 'sampler_peak_rss_bytes')):
                    try:
                        if pid is not None:
                            state[key] = max(state[key], memory(pid)['VmHWM'])
                    except (OSError, RuntimeError):
                        pass  # An already exited sampler is represented by its explicit query/disconnect error.
                state['peak_rss_bytes'] = state['supervisor_peak_rss_bytes'] + state['sampler_peak_rss_bytes']
                if state['peak_rss_bytes'] > MEMORY_LIMIT and not any(x['error_kind'] == 'memory_budget' for x in state['errors']):
                    error('memory_budget', 'Combined monitor processes exceeded 64 MiB')
                try:
                    state['sampler_returncode'], state['sampler_shutdown_action'] = stop_sampler(child)
                    state['sampler_stopped'] = child is not None and child.poll() is not None
                    expected_exit = ((state['sampler_shutdown_action'] == 'terminated' and state['sampler_returncode'] == -signal.SIGTERM)
                                     or (state['sampler_shutdown_action'] == 'killed' and state['sampler_returncode'] == -signal.SIGKILL))
                    if not expected_exit and not state['errors']:
                        error('unexpected_sampler_exit', 'Sampler exited before or during controlled shutdown: %s, %s' %
                              (state['sampler_shutdown_action'], state['sampler_returncode']))
                except Exception as exc:
                    state['sampler_stopped'] = False
                    error('shutdown_error', type(exc).__name__ + ': ' + str(exc))
                state.update(finished_unix=time.time(), complete=state.get('sampler_stopped', False),
                             passed=state['samples'] > 0 and not state['errors'] and state.get('sampler_stopped', False),
                             memory_note='peak_rss_bytes sums both processes VmHWM; monitor_rss_bytes sums their sampled RSS. No pre-exec inherited getrusage RSS is used.')
                if not (output/'ready.json').exists():
                    write(output/'ready.json', dict(pid=os.getpid(), parent_pid=controller_pid, passed=False,
                          sampler_pid=state['sampler_pid'], backend=expected_backend, errors=state['errors']))
                write(output/'summary.json', state)
                if child is not None:
                    child.stdin.close()
                    child.stdout.close()
    finally:
        if child is not None and child.poll() is None:
            stop_sampler(child)
    return 0 if state['passed'] else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--period', type=float, default=.5)
    parser.add_argument('--query-timeout', type=float, default=5.)
    parser.add_argument('--sampler', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sampler:
        return sampler_main(args.gpu)
    if not args.output:
        parser.error('--output is required')
    return run_monitor(args.output, args.gpu, args.period, args.query_timeout)


if __name__ == '__main__':
    sys.exit(main())
