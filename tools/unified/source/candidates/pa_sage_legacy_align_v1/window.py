"""Homogeneous legacy lookahead/accumulator, with explicit EOF draining.

No framework imports: exercise empty, short, merged and tail cases on CPU.
The callbacks supply real DGL samples, BaM window hints and merged SSD reads.
"""
from collections import deque
import time


class WindowAccumulator:
    def __init__(self, iterator, read_many, hint, required_accesses,
                 window_size=2, row_count=lambda batch: len(batch[0])):
        if window_size < 0 or required_accesses < 0:
            raise ValueError('Invalid lookahead or accumulator threshold')
        self.iterator = iter(iterator)
        self.read_many, self.hint, self.row_count = read_many, hint, row_count
        self.required_accesses = required_accesses
        self.window_size = window_size
        self.pending, self.ready = deque(), deque()
        self.initialized = self.exhausted = False
        self.previous_cpu_per_batch = 0
        self.sampled = self.delivered = self.read_batches = self.read_calls = 0
        self.sample_ns = self.hint_ns = 0
        self.max_merged_batches = self.max_pending = 0

    def _pull(self):
        if self.exhausted:
            return False
        start = time.perf_counter_ns()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            return False
        finally:
            self.sample_ns += time.perf_counter_ns() - start
        self.sampled += 1
        self.pending.append(batch)
        self.max_pending = max(self.max_pending, len(self.pending))
        start = time.perf_counter_ns()
        self.hint(batch)
        self.hint_ns += time.perf_counter_ns() - start
        return True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            for _ in range(self.window_size):
                if not self._pull():
                    break
            self.initialized = True
        # Preserve legacy lookahead advancement even when features are ready.
        self._pull()
        if not self.pending:
            if self.ready:
                raise RuntimeError('Features outlived their sampled batches')
            raise StopIteration
        if not self.ready:
            count, accesses = 0, 0
            while True:
                if count == len(self.pending) and not self._pull():
                    break
                accesses += self.row_count(self.pending[count])
                count += 1
                if accesses > self.required_accesses + self.previous_cpu_per_batch * count:
                    break
            batches = [self.pending[i] for i in range(count)]
            features, cpu_accesses = self.read_many(batches)
            if len(features) != count or count == 0 or cpu_accesses < 0:
                raise RuntimeError('Invalid accumulator read result')
            self.previous_cpu_per_batch = int(cpu_accesses / count)
            self.ready.extend(features)
            self.read_calls += 1
            self.read_batches += count
            self.max_merged_batches = max(self.max_merged_batches, count)
        self.delivered += 1
        return self.pending.popleft(), self.ready.popleft()

    def summary(self):
        return dict(sampled_batches=self.sampled, delivered_batches=self.delivered,
                    read_batches=self.read_batches, read_calls=self.read_calls,
                    max_merged_batches=self.max_merged_batches,
                    max_pending_batches=self.max_pending, sample_ns=self.sample_ns,
                    window_hint_ns=self.hint_ns, exhausted=self.exhausted,
                    pending_batches=len(self.pending), pending_features=len(self.ready),
                    drained=self.exhausted and not self.pending and not self.ready)
