"""Keep the accepted overlap scheduler, with an optional paired native read."""
import time
import torch
from candidates.pa_sage_512b_overlap_v1.loader import OverlapGIDS


class PairOverlapGIDS(OverlapGIDS):
    pair_enabled=False

    def _submit(self,batches):
        if not self.pair_enabled:return super()._submit(batches)
        if self.live_ticket is not None:raise RuntimeError('I/O request already pending')
        indices=[self.resolve_batch_feature_rows(b) for b in batches]
        flags=[b[2][0].srcdata['digit_storage_is_group'].to(device=self.gids_device,dtype=torch.bool).contiguous() for b in batches]
        if any(len(i)!=len(f) for i,f in zip(indices,flags)):raise RuntimeError('Group flags do not match feature rows')
        capacities=[1 << (max(2,2*len(i))-1).bit_length() for i in indices]
        tables=[torch.empty(c,dtype=torch.int32,device=self.gids_device) for c in capacities]
        outputs=[torch.empty((len(i),128),dtype=torch.float32,device=self.gids_device) for i in indices]
        self.io_stream.wait_stream(torch.cuda.current_stream(self.gids_device))
        start=time.perf_counter_ns()
        native=self.BAM_FS.read_feature_pair_async(len(batches),[x.data_ptr() for x in outputs],
            [i.data_ptr() for i in indices],[len(i) for i in indices],[f.data_ptr() for f in flags],
            [t.data_ptr() for t in tables],capacities,self.io_stream.cuda_stream)
        self.read_host_ns+=time.perf_counter_ns()-start
        ticket=dict(native=native,indices=indices,outputs=outputs,batches=batches,hints=self.hint_refs,
                    flags=flags,tables=tables)
        self.hint_refs=[];self.live_ticket=ticket;self.submitted_reads+=1
        self.last_read=dict(batches=len(batches),rows=sum(len(i) for i in indices),feature_bytes=sum(x.numel()*4 for x in outputs))
        return ticket
