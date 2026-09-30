"""Host-only nested spans. No CUDA events, device queries or synchronization."""
from contextlib import contextmanager
from functools import wraps
import importlib
import math
import time


class HostProfile:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.stack = []
        self.records = {}
        self.restores = []
        self.layer = None

    @contextmanager
    def span(self, name):
        start = self.clock()
        frame = [name, 0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - start
            assert self.stack.pop() is frame
            path = '/'.join([f[0] for f in self.stack] + [name])
            self.records.setdefault(path, []).append((elapsed, elapsed - frame[1]))
            if self.stack:
                self.stack[-1][1] += elapsed

    def replace(self, obj, name, value):
        # Restore instance attributes by deleting overrides, not retaining bound methods.
        local = name in vars(obj)
        original = getattr(obj, name)
        self.restores.append((obj, name, original, local))
        setattr(obj, name, value)

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        @wraps(original)
        def measured(*args, **kwargs):
            with self.span(label):
                return original(*args, **kwargs)
        self.replace(obj, name, measured)

    def close(self):
        if self.stack:
            raise RuntimeError('Unclosed profile spans')
        for obj, name, original, local in reversed(self.restores):
            if local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.restores.clear()

    def summary(self):
        if self.stack:
            raise RuntimeError('Cannot report incomplete spans')
        spans = {}
        for name, values in self.records.items():
            inclusive = [a for a, _ in values]
            ordered = sorted(inclusive)
            spans[name] = dict(calls=len(values), inclusive_seconds=sum(inclusive)/1e9,
                exclusive_seconds=sum(b for _, b in values)/1e9,
                first_seconds=inclusive[0]/1e9,
                subsequent_seconds=sum(inclusive[1:])/1e9,
                p50_us=ordered[(len(ordered)-1)//2]/1000,
                p95_us=ordered[math.ceil(len(ordered)*.95)-1]/1000,
                max_us=max(ordered)/1000)
        return dict(mode='host', spans=spans, added_cuda_synchronization=False,
            cuda_events=False, standalone_cuda_kernel_times=False,
            exclusive_note='Parent minus measured children; Python timing overhead remains',
            native_call_note='Host CUDA dispatch only; existing downstream waits remain at their original sites')

