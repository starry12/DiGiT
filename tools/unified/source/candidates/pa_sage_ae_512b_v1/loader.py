"""Single-batch native reads, collected before model compute in both arms."""
import time
import torch
from GIDS import GIDS
class SynchronousPairGIDS(GIDS):
    def prepare_reads(self,paired):
        if self.window_buffering_flag or self.accumulator_flag or self.page_size!=512:
            raise RuntimeError('Require 512B and no window/accumulator')
        self.pair_enabled=bool(paired)
        self.io_stream=torch.cuda.Stream(device=self.gids_device)
        self.BAM_FS.prepare_async_reads(self.io_stream.cuda_stream)
    def fetch_feature(self,dim,iterator,device):
        if dim!=128:raise ValueError('PA features must have 128 float32 values')
        began=time.perf_counter();batch=next(iterator);self.sample_time+=time.perf_counter()-began
        began=time.perf_counter();ids=self.resolve_batch_feature_rows(batch)
        out=torch.empty((len(ids),128),dtype=torch.float32,device=self.gids_device)
        self.io_stream.wait_stream(torch.cuda.current_stream(self.gids_device))
        flags=table=None
        if self.pair_enabled:
            flags=batch[2][0].srcdata['digit_storage_is_group'].to(device=self.gids_device,dtype=torch.bool).contiguous()
            if flags.shape!=ids.shape:raise ValueError('Mismatched group flags')
            capacity=1<<(max(2,2*len(ids))-1).bit_length()
            table=torch.empty(capacity,dtype=torch.int32,device=self.gids_device)
            self.io_stream.wait_stream(torch.cuda.current_stream(self.gids_device))
        try:
            if self.pair_enabled:
                ticket=self.BAM_FS.read_feature_pair_async(1,[out.data_ptr()],[ids.data_ptr()],[len(ids)],
                    [flags.data_ptr()],[table.data_ptr()],[capacity],self.io_stream.cuda_stream)
            else:
                ticket=self.BAM_FS.read_feature_merged_async(1,[out.data_ptr()],[ids.data_ptr()],[len(ids)],128,128,self.io_stream.cuda_stream)
            # collect blocks until output is ready. No lookahead or compute overlap;
            # ids/flags/table/output stay referenced through native completion.
            self.BAM_FS.collect_feature_read(ticket)
        except BaseException:
            self.BAM_FS.synchronize_async_reads()
            raise
        self.feature_time+=time.perf_counter()-began
        return (*batch,out)
