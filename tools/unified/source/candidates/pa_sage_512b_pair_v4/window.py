"""Bound lookahead/accumulation without changing sampling, root batches or EOF."""
from collections import deque
import time

MAX_BATCHES=4
MAX_ROWS=262144


class WindowAccumulator:
    def __init__(self,iterator,read_many,hint,required_accesses,window_size=2,
                 row_count=lambda b:len(b[0]),max_batches=MAX_BATCHES,max_rows=MAX_ROWS):
        if window_size<0 or required_accesses<0 or max_batches<1 or max_rows<1:
            raise ValueError('Invalid lookahead/accumulator bounds')
        self.iterator=iter(iterator);self.read_many=read_many;self.hint=hint;self.row_count=row_count
        self.required_accesses=required_accesses;self.window_size=window_size
        self.max_batches=max_batches;self.max_rows=max_rows
        self.pending=deque();self.ready=deque();self.initialized=self.exhausted=False
        self.previous_cpu_per_batch=0
        self.sampled=self.delivered=self.read_batches=self.read_calls=0
        self.sample_ns=self.hint_ns=0
        self.max_merged_batches=self.max_pending=self.max_read_rows=0
        self.batch_cap_flushes=self.row_cap_flushes=self.oversized_single_batches=0
        self.current_read=None

    def _pull(self):
        if self.exhausted:return False
        start=time.perf_counter_ns()
        try:batch=next(self.iterator)
        except StopIteration:
            self.exhausted=True;return False
        finally:self.sample_ns+=time.perf_counter_ns()-start
        self.pending.append(batch);self.sampled+=1
        self.max_pending=max(self.max_pending,len(self.pending))
        start=time.perf_counter_ns();self.hint(batch);self.hint_ns+=time.perf_counter_ns()-start
        return True

    def __iter__(self):return self

    def __next__(self):
        if not self.initialized:
            for _ in range(self.window_size):
                if not self._pull():break
            self.initialized=True
        self._pull()
        if not self.pending:
            if self.ready:raise RuntimeError('Features outlived sampled batches')
            raise StopIteration
        if not self.ready:
            count=accesses=0;reason='eof'
            while True:
                if count>=self.max_batches:
                    self.batch_cap_flushes+=1;reason='batch_cap';break
                if count==len(self.pending) and not self._pull():break
                rows=self.row_count(self.pending[count])
                if count and accesses+rows>self.max_rows:
                    self.row_cap_flushes+=1;reason='row_cap';break
                accesses+=rows;count+=1
                # Never split/drop a root batch to enforce a feature row budget.
                if count==1 and rows>self.max_rows:
                    self.oversized_single_batches+=1;reason='oversized_single';break
                if accesses>self.required_accesses+self.previous_cpu_per_batch*count:
                    reason='predicted_io';break
            self.current_read=dict(batches=count,rows=accesses,reason=reason)
            features,cpu_accesses=self.read_many([self.pending[i] for i in range(count)])
            if count==0 or len(features)!=count or not 0<=cpu_accesses<=accesses:
                raise RuntimeError('Invalid accumulator read result')
            self.previous_cpu_per_batch=int(cpu_accesses/count)
            self.ready.extend(features);self.read_calls+=1;self.read_batches+=count
            self.max_merged_batches=max(self.max_merged_batches,count)
            self.max_read_rows=max(self.max_read_rows,accesses)
        self.delivered+=1
        return self.pending.popleft(),self.ready.popleft()

    def summary(self):
        return dict(sampled_batches=self.sampled,delivered_batches=self.delivered,read_batches=self.read_batches,
            read_calls=self.read_calls,max_merged_batches=self.max_merged_batches,max_pending_batches=self.max_pending,
            max_read_rows=self.max_read_rows,limit_batches=self.max_batches,limit_rows=self.max_rows,
            batch_cap_flushes=self.batch_cap_flushes,row_cap_flushes=self.row_cap_flushes,
            oversized_single_batches=self.oversized_single_batches,current_read=self.current_read,
            sample_ns=self.sample_ns,window_hint_ns=self.hint_ns,exhausted=self.exhausted,
            pending_batches=len(self.pending),pending_features=len(self.ready),
            drained=self.exhausted and not self.pending and not self.ready)
