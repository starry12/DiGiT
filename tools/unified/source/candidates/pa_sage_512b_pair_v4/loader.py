"""Adapt only the homogeneous window/accumulator path; keep native reads."""
import time
import torch
from GIDS import GIDS
from .window import WindowAccumulator


class BoundedGIDS(GIDS):
    def begin_epoch(self):
        self.pipeline = None
        self.pipeline_iterator = None
        self.fetch_host_ns = self.read_host_ns = 0
        self.delivered_rows = 0
        self.last_read = None
        self.sample_time = self.feature_time = self.GIDS_time = self.WB_time = 0.0

    def fetch_feature(self, dim, iterator, device):
        if not self.accumulator_flag or not self.window_buffering_flag or self.heterograph:
            raise RuntimeError('Legacy alignment requires homogeneous WB + accumulator')
        if self.pipeline is None:
            self.pipeline_iterator = iterator
            self.pipeline = WindowAccumulator(iterator, self._read_many,
                self.window_buffering, self.required_accesses, self.wb_size)
        if iterator is not self.pipeline_iterator:
            raise RuntimeError('Call begin_epoch before changing the source iterator')
        start = time.perf_counter_ns()
        try:
            batch, features = next(self.pipeline)
        finally:
            self.fetch_host_ns += time.perf_counter_ns() - start
            self.sample_time = self.pipeline.sample_ns / 1e9
        self.delivered_rows += len(batch[0])
        return (*batch, features)

    def _read_many(self, batches):
        self.last_read = dict(batches=len(batches), rows=sum(len(b[0]) for b in batches),
                              feature_bytes=sum(len(b[0]) for b in batches)*self.cache_dim*4)
        indices = [self.resolve_batch_feature_rows(b) for b in batches]
        outputs = [torch.zeros((len(i), self.cache_dim), dtype=torch.float32,
                               device=self.gids_device) for i in indices]
        start = time.perf_counter_ns()
        self.BAM_FS.read_feature_merged(len(batches), [x.data_ptr() for x in outputs],
            [i.data_ptr() for i in indices], [len(i) for i in indices],
            self.cache_dim, self.cache_dim)
        self.read_host_ns += time.perf_counter_ns() - start
        self.feature_time = self.read_host_ns / 1e9
        cpu = self.BAM_FS.get_cpu_access_count()
        self.BAM_FS.flush_cpu_access_count()
        return outputs, cpu
