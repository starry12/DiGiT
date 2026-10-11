"""Tiny real CUDA stream/owner test; no NVMe controller or graph allocation."""
def gpu_fixture():
    import torch
    from .pipeline import Reader
    class Features:
        def fetch_device(self,rows,flags,asynchronous=False):
            assert asynchronous
            # Actual asynchronous work on Reader's private stream.
            return rows.float().reshape(-1,1).expand(-1,1024).contiguous()
    reader=Reader(Features(),max_rows=32)
    def batches():
        for i in range(3):
            rows=torch.arange(1+i*5,6+i*5,device='cuda',dtype=torch.int64)
            yield (None,None,None,rows,torch.ones_like(rows,dtype=torch.bool),None)
    try:
        for batch,x in reader.window(batches()):
            assert torch.equal(x[:,0].long(),batch[3])
            assert torch.equal(x.sum(1),batch[3].float()*1024)
        assert reader.live is None and reader.submitted==reader.collected==3
        reader.submit(next(batches()))
        try:reader.submit(next(batches()))
        except RuntimeError:pass
        else:raise AssertionError('Outstanding submit accepted')
        reader.collect()
    finally:reader.close()
    return 4
