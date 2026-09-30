"""Non-overlapping synchronized wall spans, separate from performance runs."""
import time
from contextlib import contextmanager
class StageTimer:
    def __init__(self,synchronize,enabled,clock=time.perf_counter):
        self.synchronize=synchronize;self.enabled=enabled;self.clock=clock;self.reset()
    def reset(self):self.values={};self.active=False
    @contextmanager
    def stage(self,name):
        if not self.enabled:yield;return
        if self.active:raise ValueError('Nested spans would double-count time')
        self.active=True
        try:
            self.synchronize();start=self.clock()
            try:yield
            finally:
                self.synchronize();elapsed=self.clock()-start
                row=self.values.setdefault(name,dict(seconds=0.,calls=0));row['seconds']+=elapsed;row['calls']+=1
        finally:self.active=False
    def report(self):return {k:dict(v) for k,v in self.values.items()}
