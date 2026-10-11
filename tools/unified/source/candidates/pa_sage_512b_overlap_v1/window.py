"""Split v4's bounded window into preparation and feature acquisition.

Callbacks own all device work: submit_read(batches) returns an opaque ticket;
collect_read(ticket) waits and returns (features, cpu_accesses).  No second read
is submitted before the first ticket is collected.  A prepared batch stays in
pending until acquire_next(), including when its features were already ready.

With WB2 and a four-batch accumulator, at most four sampled batches are retained
here, plus the current training batch held by the caller.  The row limit bounds
each merged read, not all lookahead batches; an oversized root batch is retained
intact and read alone.  Callbacks must keep submitted buffers alive until collect.
After any callback failure the scheduler is terminal and preserves the ticket
reference: the caller must finish/cancel device work before releasing buffers.
"""
from collections import deque
import time


MAX_BATCHES = 4
MAX_ROWS = 262144
_NO_TICKET = object()


class WindowAccumulator:
    def __init__(self, iterator, submit_read, collect_read, hint,
                 required_accesses, window_size=2,
                 row_count=lambda b: len(b[0]), max_batches=MAX_BATCHES,
                 max_rows=MAX_ROWS):
        if window_size < 0 or required_accesses < 0 or max_batches < 1 or max_rows < 1:
            raise ValueError('Invalid lookahead/accumulator bounds')
        self.iterator = iter(iterator)
        self.submit_read = submit_read
        self.collect_read = collect_read
        self.hint = hint
        self.row_count = row_count
        self.required_accesses = required_accesses
        self.window_size = window_size
        self.max_batches = max_batches
        self.max_rows = max_rows
        self.pending = deque()
        self.ready = deque()
        self.initialized = self.exhausted = False
        self.previous_cpu_per_batch = 0
        self.sampled = self.delivered = self.read_batches = self.read_calls = 0
        self.sample_ns = self.hint_ns = 0
        self.max_merged_batches = self.max_pending = self.max_read_rows = 0
        self.batch_cap_flushes = self.row_cap_flushes = self.oversized_single_batches = 0
        self.current_read = None
        self._ticket = _NO_TICKET
        self._prepared = self._finished = False
        self._failure = None
        self.async_submit_calls = self.async_collect_calls = 0
        self.submitted_reads = 0
        self.max_outstanding_reads = self.max_ready_features = 0

    @property
    def has_outstanding_read(self):
        return self._ticket is not _NO_TICKET

    @property
    def outstanding_ticket(self):
        """Retain an opaque ticket for caller-owned cleanup after a failure."""
        return self._ticket if self.has_outstanding_read else None

    def _check_failure(self):
        if self._failure is not None:
            raise RuntimeError('Window accumulator failed; cannot retry callbacks') from self._failure

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
        self.pending.append(batch)
        self.sampled += 1
        self.max_pending = max(self.max_pending, len(self.pending))
        start = time.perf_counter_ns()
        self.hint(batch)
        self.hint_ns += time.perf_counter_ns() - start
        return True

    def __iter__(self):
        return self

    def prepare_next(self):
        """Prepare one delivery without waiting for a submitted read.

        Repeated calls before acquire are idempotent.  False means the iterator
        has fully drained; it does not mean that a ticket is merely unfinished.
        """
        self._check_failure()
        if self._prepared:
            return True
        if self._finished:
            return False
        try:
            if not self.initialized:
                for _ in range(self.window_size):
                    if not self._pull():
                        break
                self.initialized = True
            self._pull()
            if not self.pending:
                if self.ready or self.has_outstanding_read:
                    raise RuntimeError('Features outlived sampled batches')
                self._finished = True
                return False
            if not self.ready:
                count = accesses = 0
                reason = 'eof'
                while True:
                    if count >= self.max_batches:
                        self.batch_cap_flushes += 1
                        reason = 'batch_cap'
                        break
                    if count == len(self.pending) and not self._pull():
                        break
                    rows = self.row_count(self.pending[count])
                    if count and accesses + rows > self.max_rows:
                        self.row_cap_flushes += 1
                        reason = 'row_cap'
                        break
                    accesses += rows
                    count += 1
                    if count == 1 and rows > self.max_rows:
                        self.oversized_single_batches += 1
                        reason = 'oversized_single'
                        break
                    if accesses > self.required_accesses + self.previous_cpu_per_batch * count:
                        reason = 'predicted_io'
                        break
                self.current_read = dict(batches=count, rows=accesses, reason=reason)
                self.async_submit_calls += 1
                self._ticket = self.submit_read([self.pending[i] for i in range(count)])
                self.submitted_reads += 1
                self.max_outstanding_reads = max(self.max_outstanding_reads, 1)
            self._prepared = True
            return True
        except StopIteration as error:
            self._failure = RuntimeError('Preparation callback raised StopIteration')
            raise self._failure from error
        except BaseException as error:
            self._failure = error
            raise

    def acquire_next(self):
        """Collect a prepared read, then deliver exactly one batch/features pair."""
        self._check_failure()
        if self._finished:
            raise StopIteration
        if not self._prepared:
            raise RuntimeError('prepare_next() must precede acquire_next()')
        try:
            if self.has_outstanding_read:
                count = self.current_read['batches']
                accesses = self.current_read['rows']
                self.async_collect_calls += 1
                features, cpu_accesses = self.collect_read(self._ticket)
                if count == 0 or len(features) != count or not 0 <= cpu_accesses <= accesses:
                    raise RuntimeError('Invalid accumulator read result')
                self.previous_cpu_per_batch = int(cpu_accesses / count)
                self.ready.extend(features)
                self.read_calls += 1
                self.read_batches += count
                self.max_merged_batches = max(self.max_merged_batches, count)
                self.max_read_rows = max(self.max_read_rows, accesses)
                self.max_ready_features = max(self.max_ready_features, len(self.ready))
                self._ticket = _NO_TICKET
            batch, features = self.pending.popleft(), self.ready.popleft()
            self.delivered += 1
            self._prepared = False
            return batch, features
        except StopIteration as error:
            self._failure = RuntimeError('Read callback raised StopIteration')
            raise self._failure from error
        except BaseException as error:
            self._failure = error
            raise

    def __next__(self):
        if not self.prepare_next():
            raise StopIteration
        return self.acquire_next()

    def summary(self):
        return dict(
            sampled_batches=self.sampled, delivered_batches=self.delivered,
            read_batches=self.read_batches, read_calls=self.read_calls,
            max_merged_batches=self.max_merged_batches, max_pending_batches=self.max_pending,
            max_read_rows=self.max_read_rows, limit_batches=self.max_batches,
            limit_rows=self.max_rows, batch_cap_flushes=self.batch_cap_flushes,
            row_cap_flushes=self.row_cap_flushes,
            oversized_single_batches=self.oversized_single_batches,
            current_read=self.current_read, sample_ns=self.sample_ns,
            window_hint_ns=self.hint_ns, exhausted=self.exhausted,
            pending_batches=len(self.pending), pending_features=len(self.ready),
            drained=(self.exhausted and not self.pending and not self.ready
                     and not self.has_outstanding_read and not self._prepared
                     and self._failure is None),
            async_submit_calls=self.async_submit_calls,
            async_collect_calls=self.async_collect_calls,
            submitted_reads=self.submitted_reads,
            collected_reads=self.read_calls,
            outstanding_reads=int(self.has_outstanding_read),
            max_outstanding_reads=self.max_outstanding_reads,
            prepared_batches=int(self._prepared),
            max_ready_features=self.max_ready_features,
            retained_batches=len(self.pending),
            retained_batch_limit=max(self.window_size + 1, self.max_batches),
            inflight_batches=(self.current_read['batches'] if self.has_outstanding_read else 0),
            failed=self._failure is not None,
        )
