"""One outstanding bounded read; retain DMA owners until stream completion."""
import torch
class Reader:
    def __init__(self,features,max_rows=405504):
        self.features=features;self.max_rows=max_rows;self.stream=torch.cuda.Stream()
        self.live=None;self.submitted=self.collected=0
    def submit(self,batch):
        if self.live is not None:raise RuntimeError('Outstanding IG read')
        inputs,outputs,blocks,rows,flags,targets=batch
        if not 0<len(rows)<=self.max_rows:raise ValueError('IG frontier bound')
        self.stream.wait_stream(torch.cuda.current_stream())
        self.live=dict(batch=batch,output=None)
        with torch.cuda.stream(self.stream):
            self.live['output']=self.features.fetch_device(rows,flags,asynchronous=True)
            for v in (rows,flags,self.live['output']):v.record_stream(self.stream)
        self.submitted+=1
    def collect(self):
        if self.live is None or self.live['output'] is None:raise RuntimeError('Incomplete native submission')
        self.stream.synchronize() # Collect asynchronous kernel errors before reuse.
        current=torch.cuda.current_stream();current.wait_stream(self.stream)
        row=self.live;row['output'].record_stream(current);self.live=None;self.collected+=1
        return row['batch'],row['output']
    def window(self,batches):
        it=iter(batches);self.submit(next(it));current=self.collect()
        while True:
            try:following=next(it)
            except StopIteration:yield current;break
            self.submit(following);yield current;current=self.collect()
        if self.live is not None:raise RuntimeError('Undrained window')
    def close(self):
        # A synchronization failure keeps live owners attached to this reader.
        self.stream.synchronize();self.live=None
