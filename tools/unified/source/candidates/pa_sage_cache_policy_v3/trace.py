"""Compact GPU workload checksums; no per-batch feature/edge transfer to CPU."""
import hashlib
import json


class Trace:
    def __init__(self):
        self.header=hashlib.sha256();self.parts=[]

    def add(self,label,tensor):
        import torch
        v=tensor.detach().reshape(-1).to(torch.int64)
        self.header.update(json.dumps([label,list(tensor.shape),str(tensor.dtype)]).encode())
        position=torch.arange(1,v.numel()+1,device=v.device,dtype=torch.int64)
        first=(v*position).sum()
        second=((v^(v<<13))*(position*2654435761+2246822519)).sum()
        self.parts.append(torch.stack((first,second)))

    def digest(self):
        import torch
        h=self.header.copy()
        if self.parts:h.update(torch.stack(self.parts).cpu().numpy().tobytes())
        return h.hexdigest()


def add_batch(sample,storage,batch,inp,out,blocks):
    import dgl
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP
    sample.add((batch,'inputs'),inp);sample.add((batch,'outputs'),out)
    for i,block in enumerate(blocks):
        u,v=block.edges(order='eid')
        for key,tensor in [('src',block.srcdata[dgl.NID]),('dst',block.dstdata[dgl.NID]),
                           ('u',u),('v',v),('eid',block.edata[dgl.EID])]:
            sample.add((batch,i,key),tensor)
    storage.add((batch,'rows'),blocks[0].srcdata[DIGIT_STORAGE_ROW])
    storage.add((batch,'flags'),blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP])
