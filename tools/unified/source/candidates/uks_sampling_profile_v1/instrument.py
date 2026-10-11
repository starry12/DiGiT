"""Nested host spans plus non-synchronizing events on the original stream."""
from .host_profile import HostProfile
from .common import require,extension
class Probe:
    def __init__(self,sampler,enabled):
        self.sampler=sampler;self.enabled=enabled;self.host=HostProfile();self.batch=0;self.wall=[]
        import torch
        self.torch=torch
        if enabled:
            from . import sampler as instrumented
            self.native=extension();instrumented._cuda_extension=self.native
            sampler.__class__=instrumented.ProfiledDiGiTNeighborSampler;sampler.profile=self.host
            for name,label in [('sample_inner_frontier','inner_dgl_sampling'),('build_block','block_build'),('annotate_outer_storage','storage_annotation'),('_attach_cuda_storage_rows','address_mapping')]:self.host.wrap(sampler,name,label)
            self.native.profile_prepare(300,torch.cuda.current_stream().cuda_stream)
        old=sampler.sample_blocks
        def sample(*args,**kwargs):
            self.batch+=1
            if not self.enabled:return old(*args,**kwargs)
            if self.batch==21:self.host.records.clear()
            if self.batch>20:self.native.profile_arm(self.batch)
            with self.host.span('sample_blocks'):return old(*args,**kwargs)
        sampler.sample_blocks=sample
    def finish(self):
        require(self.batch==320,'Incomplete sampling profile')
        if not self.enabled:return dict(enabled=False,calls=320)
        self.torch.cuda.synchronize();events=[dict(r) for r in self.native.profile_collect()];self.native.profile_release()
        require(len(events)==300 and [r['batch'] for r in events]==list(range(21,321)),'Incomplete native events')
        for r in events:require(abs(r['total_ms']-r['group_sample_ms']-r['resolve_eids_ms'])<.05,'Event accounting mismatch')
        host=self.host.summary();self.host.close()
        spans=host['spans'];require(spans['sample_blocks']['calls']==300 and spans['sample_blocks/block_build']['calls']==900 and spans['sample_blocks/storage_annotation/address_mapping']['calls']==300,'Missing host spans')
        return dict(enabled=True,native_events=events,host=host,group_sample_seconds=sum(r['group_sample_ms'] for r in events)/1000,eid_resolution_seconds=sum(r['resolve_eids_ms'] for r in events)/1000,native_stream_seconds=sum(r['total_ms'] for r in events)/1000,event_note='Same-stream kernel brackets; may include device scheduling and launch gaps, not pure active instructions',host_note='Nested host inclusive/exclusive spans; dispatch span is not CUDA execution time')
