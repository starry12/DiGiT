"""Process-local affinity and low-frequency host observations only."""
import ctypes
import os
from pathlib import Path

LIBC = ctypes.CDLL(None)
LIBC.sched_getcpu.restype = ctypes.c_int


def thread_masks():
    result = {}
    for task in Path('/proc/self/task').iterdir():
        try:
            result[task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            pass
    return result


def apply(policy, cpu):
    if policy not in ('unbound', 'one_core'):
        raise ValueError('Unknown affinity policy')
    before = thread_masks()
    if policy == 'one_core':
        if cpu not in os.sched_getaffinity(0):
            raise RuntimeError('Requested logical CPU is outside the worker CPU set')
        # Later native threads inherit the single-CPU mask from their creators.
        for tid in before:
            try:
                os.sched_setaffinity(int(tid), {cpu})
            except ProcessLookupError:
                pass
    return dict(policy=policy, cpu=cpu, before=before, at_training_start=thread_masks())


def observation(update, cpu):
    current = LIBC.sched_getcpu()
    def frequency(which):
        return int(Path('/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq' % which).read_text())
    return dict(update=update, main_logical_cpu=current,
                main_frequency_khz=frequency(current), target_cpu=cpu, target_frequency_khz=frequency(cpu))
