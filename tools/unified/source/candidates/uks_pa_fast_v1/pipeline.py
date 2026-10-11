"""One outstanding I/O batch, no WB hints, no accumulator, no extra thread."""
import torch
class Reader:
    def __init__(self,loader,max_rows):
        self.loader=loader;self.fs=loader.BAM_FS;self.stream=torch.cuda.Stream();self.live=None
        self.submitted=0;self.collected=0;self.max_rows=max_rows
        self.fs.prepare_policy_async(self.stream.cuda_stream,max_rows)
    def submit(self,batch):
        if self.live is not None:raise RuntimeError('Overlapping I/O ownership')
        ids,flags=self.loader.resolve_batch_feature_accesses(batch)
        if not 0<len(ids)<=self.max_rows or len(ids)!=len(flags):raise RuntimeError('Unbounded/mismatched read')
        output=torch.empty((len(ids),256),device=ids.device,dtype=torch.float32)
        self.stream.wait_stream(torch.cuda.current_stream())
        # Keep owners before submission so exceptions also retain DMA pointers.
        self.live=dict(batch=batch,ids=ids,flags=flags,output=output,ticket=None)
        self.live['ticket']=self.fs.submit_policy_async(output.data_ptr(),ids.data_ptr(),flags.data_ptr(),len(ids),self.stream.cuda_stream)
        self.submitted+=1
    def collect(self):
        if self.live is None or self.live['ticket'] is None:raise RuntimeError('Missing native ticket')
        row=self.live;self.fs.collect_policy_async(row['ticket'])
        torch.cuda.current_stream().wait_stream(self.stream)
        row['output'].record_stream(torch.cuda.current_stream())
        self.live=None;self.collected+=1
        return (*row['batch'],row['output'])
    def window(self,iterator,count):
        if count<=0 or self.live is not None:raise RuntimeError('Invalid window boundary')
        self.submit(next(iterator));current=self.collect()
        for i in range(count):
            if i+1<count:self.submit(next(iterator))
            yield current
            if i+1<count:current=self.collect()
        if self.live is not None:raise RuntimeError('Window not drained')
    def close(self):
        # Native close synchronizes the private stream even after partial submit.
        self.fs.close_policy_async();self.stream.synchronize();self.live=None
    def summary(self):
        return dict(submitted=self.submitted,collected=self.collected,outstanding=int(self.live is not None),
            max_prefetch_batches=1,max_rows=self.max_rows,additional_feature_buffer_bound_bytes=self.max_rows*1024,
            pair_hash_capacity=1<<(2*self.max_rows-1).bit_length(),window_buffer=False,accumulator=False,additional_sampling_threads=0)
