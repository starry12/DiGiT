"""20 warmup + 300 measured batches; value checks only during warmup."""
import numpy as np
import torch
from .training import Engine as Base
from . import protocol as P
from candidates.ukl_postprocess_v22.request import request_from_blocks as optimized_request
from candidates.ukl_training_adapter_v12.adapter import request_from_blocks as baseline_request
from candidates.ukl_runtime_prepare_v11r1.data import features

class Engine(Base):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.feature_checks=[];self.request_from_blocks=optimized_request
    def window(self,*args,**kwargs):
        result=super().window(*args,**kwargs)
        from .timing import TIMING
        result["timing_protocol"]=dict(TIMING)
        return result

    def step(self,roots,labels):
        if self.updates>=P.TOTAL:raise RuntimeError('UKL 320-batch performance cap')
        self.check();roots=np.asarray(roots);labels=np.asarray(labels)
        if roots.shape!=(1024,) or labels.shape!=roots.shape or labels.dtype!=np.int64 or np.any(labels<0) or np.any(labels>=19):raise ValueError('Performance roots/labels shape')
        inp,out,blocks=self.sampler.sample_blocks(roots,self.updates)
        req=self.request_from_blocks(self.arm,self.native.graph,inp,out,blocks,roots)
        x=self.provider(req)
        if x.shape!=(len(inp),128) or x.dtype!=torch.float32 or x.device!=self.device or x.requires_grad:raise ValueError('Feature tensor contract')
        if self.updates in (0,1,2,3):
            # Only 16 KiB CPU reference, no additional SSD requests or persistent warming.
            pick=np.linspace(0,len(inp)-1,32,dtype=np.int64)
            actual=x[torch.as_tensor(pick,device=self.device)].detach().cpu().numpy()
            if not np.array_equal(actual,features(req.logical_ids[pick])):raise RuntimeError('Actual SSD/cache feature values differ from logical source')
            self.feature_checks.append(dict(passed=True,rows=32,dim=128,batch=self.updates))
        blocks=[b.to(self.device) for b in blocks];target=torch.from_numpy(labels.copy()).to(self.device)
        self.model.train();loss=torch.nn.functional.cross_entropy(self.model(blocks,x),target)
        self.opt.zero_grad(set_to_none=True);loss.backward();self.check();self.opt.step();self.updates+=1
        return loss.detach(),len(inp)
