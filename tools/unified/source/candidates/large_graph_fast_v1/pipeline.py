"""One live native ticket; all DMA owners retained through collect/close."""
import torch
from .sampling import STORAGE,HEADS

class Reader:
    def __init__(self,provider,max_rows=405504):
        self.provider=provider;self.fs=provider.fs;self.stream=torch.cuda.Stream();self.live=None
        self.max_rows=max_rows;self.submitted=self.collected=0
        self.fs.prepare_policy_async(self.stream.cuda_stream,max_rows)
    def submit(self,batch):
        if self.live is not None:raise RuntimeError('Previous prefetch not collected')
        rows,heads=self.provider.device_request(batch)
        if not 0<len(rows)<=self.max_rows:raise ValueError('Feature prefetch capacity')
        output=torch.empty((len(rows),128),dtype=torch.float32,device=rows.device)
        self.stream.wait_stream(torch.cuda.current_stream())
        self.live=dict(batch=batch,rows=rows,heads=heads,output=output,ticket=None)
        self.live['ticket']=self.fs.submit_policy_async(output.data_ptr(),rows.data_ptr(),heads.data_ptr(),len(rows),self.stream.cuda_stream)
        self.submitted+=1
    def collect(self):
        if self.live is None or self.live['ticket'] is None:raise RuntimeError('Missing prefetch ticket')
        row=self.live;self.fs.collect_policy_async(row['ticket']);torch.cuda.current_stream().wait_stream(self.stream)
        row['output'].record_stream(torch.cuda.current_stream());self.live=None;self.collected+=1
        return row['batch'],row['output']
    def window(self,batches):
        batches=iter(batches)
        try:first=next(batches)
        except StopIteration:return
        self.submit(first);current=self.collect()
        while True:
            try:following=next(batches)
            except StopIteration:
                yield current;break
            self.submit(following);yield current;current=self.collect()
    def close(self):
        self.fs.close_policy_async();self.stream.synchronize();self.live=None
    def summary(self):return dict(submitted=self.submitted,collected=self.collected,outstanding=int(self.live is not None),max_prefetch_batches=1)

class DeviceProvider:
    def device_request(self,batch):
        inp,out,blocks=batch[:3]
        if self.closed or len(blocks)!=3:raise ValueError('Invalid device feature request')
        rows=blocks[0].srcdata[STORAGE];heads=blocks[0].srcdata[HEADS]
        if not rows.is_cuda or rows.dtype!=torch.int64 or heads.dtype!=torch.int64 or rows.shape!=inp.shape or heads.shape!=rows.shape or not 0<len(rows)<=405504:
            raise ValueError('Feature request geometry')
        # Native sampler already checked EIDs, logical aliases and exact pair heads.
        return rows.contiguous(),heads.contiguous()
    def fetch_device(self,batch):
        rows,heads=self.device_request(batch);x=torch.empty((len(rows),128),device=rows.device)
        self.fs.read_feature(x.data_ptr(),rows.data_ptr(),len(rows),128,128,0,heads.data_ptr())
        return x
