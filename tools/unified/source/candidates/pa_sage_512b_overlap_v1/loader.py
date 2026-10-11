"""Single-threaded scheduling with a distinct I/O stream; no sampling worker."""
import time
import torch
from candidates.pa_sage_512b_pair_v4.loader import BoundedGIDS
from .window import WindowAccumulator


class OverlapGIDS(BoundedGIDS):
    def begin_epoch(self):
        super().begin_epoch()
        self.io_stream=torch.cuda.Stream(device=self.gids_device)
        self.BAM_FS.prepare_async_reads(self.io_stream.cuda_stream)
        self.hint_refs=[]
        self.live_ticket=None
        self.submitted_reads=self.collected_reads=0
        self.prepare_host_ns=self.collect_host_ns=0

    def bind(self,iterator):
        if self.pipeline is not None:raise RuntimeError('Only one iterator per epoch')
        self.pipeline_iterator=iterator
        self.pipeline=WindowAccumulator(iterator,self._submit,self._collect,
            self._hint,self.required_accesses,self.wb_size)

    def _hint(self,batch):
        rows=self.resolve_batch_feature_rows(batch)
        # The default stream owns sampling. This dependency never waits on future model work.
        self.io_stream.wait_stream(torch.cuda.current_stream(self.gids_device))
        self.BAM_FS.set_window_buffering_async(rows.data_ptr(),len(rows),0,self.io_stream.cuda_stream)
        self.hint_refs.append(rows)

    def _submit(self,batches):
        if self.live_ticket is not None:raise RuntimeError('I/O owner already has a read')
        indices=[self.resolve_batch_feature_rows(b) for b in batches]
        outputs=[torch.empty((len(i),self.cache_dim),dtype=torch.float32,device=self.gids_device) for i in indices]
        self.io_stream.wait_stream(torch.cuda.current_stream(self.gids_device))
        start=time.perf_counter_ns()
        native=self.BAM_FS.read_feature_merged_async(len(batches),[x.data_ptr() for x in outputs],
            [i.data_ptr() for i in indices],[len(i) for i in indices],self.cache_dim,self.cache_dim,
            self.io_stream.cuda_stream)
        self.read_host_ns+=time.perf_counter_ns()-start
        # Hold every pointer owner until the native completion event has finished.
        ticket=dict(native=native,indices=indices,outputs=outputs,batches=batches,hints=self.hint_refs)
        self.hint_refs=[]
        self.live_ticket=ticket;self.submitted_reads+=1
        self.last_read=dict(batches=len(batches),rows=sum(len(i) for i in indices),
                            feature_bytes=sum(x.numel()*x.element_size() for x in outputs))
        return ticket

    def _collect(self,ticket):
        if ticket is not self.live_ticket:raise RuntimeError('Wrong I/O ticket')
        start=time.perf_counter_ns()
        cpu=self.BAM_FS.collect_feature_read(ticket['native'])
        elapsed=time.perf_counter_ns()-start
        self.collect_host_ns+=elapsed;self.read_host_ns+=elapsed
        self.live_ticket=None;self.collected_reads+=1
        self.feature_time=self.read_host_ns/1e9
        return ticket['outputs'],cpu

    def prepare_next(self):
        start=time.perf_counter_ns()
        try:return self.pipeline.prepare_next()
        finally:
            elapsed=time.perf_counter_ns()-start
            self.prepare_host_ns+=elapsed;self.fetch_host_ns+=elapsed

    def acquire_next(self):
        from GIDS.breakdown import fetch_feature_record
        start=time.perf_counter_ns()
        batch,features=self.pipeline.acquire_next()
        elapsed=time.perf_counter_ns()-start
        self.fetch_host_ns+=elapsed
        # One fetch record per delivered batch; prepare is already counted separately.
        fetch_feature_record(elapsed)
        self.delivered_rows+=len(batch[0]);self.sample_time=self.pipeline.sample_ns/1e9
        return (*batch,features)

    def drain(self):
        # Exception/final teardown only. Never free raw-pointer owners before CUDA is done.
        if self.live_ticket is not None:self._collect(self.live_ticket)
        self.io_stream.synchronize()
        self.hint_refs=[]

    def async_summary(self):
        return dict(outstanding=int(self.live_ticket is not None),submitted=self.submitted_reads,
            collected=self.collected_reads,prepare_host_seconds=self.prepare_host_ns/1e9,
            collect_host_seconds=self.collect_host_ns/1e9,additional_sampling_threads=0)


def batches(dataloader,loader,mode):
    if mode=='legacy':
        yield from dataloader
        return
    # GIDS creates/shuffles the dataset once. Sampling still goes through its CollateWrapper.
    source=iter(dataloader).dataloader_it
    loader.bind(source)
    if not loader.prepare_next():return
    current=loader.acquire_next()
    while True:
        available=loader.prepare_next()
        following=loader.acquire_next() if available and mode=='serial' else None
        yield current
        if not available:break
        current=following if mode=='serial' else loader.acquire_next()
    loader.drain()
