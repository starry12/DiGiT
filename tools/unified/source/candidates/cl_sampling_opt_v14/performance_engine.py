"""One symmetric diagnostic pair with synchronized, nested stage timings."""
import numpy as np
import torch
from .training import Engine as Base
from . import protocol as P
from .instrument import Timer,Hooks
from candidates.ukl_postprocess_v22.request import request_from_blocks as optimized_request
from candidates.ukl_training_adapter_v12.adapter import request_from_blocks as baseline_request
from candidates.ukl_runtime_prepare_v11r1.data import features

class Engine(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feature_checks=[]
        self.request_from_blocks=baseline_request if self.arm=='gids' else optimized_request
        self.profile=Timer(torch.cuda.synchronize)

    def window(self,*args,**kwargs):
        with Hooks(self.profile,self.native,self.sampler,self.arm):
            result=super().window(*args,**kwargs)
        result['stage_profile']=self.profile.report()
        return result

    def step(self,roots,labels):
        if self.updates>=P.TOTAL:raise RuntimeError('CL 320-batch diagnostic cap')
        t=self.profile;t.active=self.updates>=P.WARMUP
        with t.measure('batch_other'):
            with t.measure('runtime_check'):self.check()
            roots=np.asarray(roots);labels=np.asarray(labels)
            if roots.shape!=(1024,) or labels.shape!=roots.shape or labels.dtype!=np.int64 or np.any(labels<0) or np.any(labels>=19):raise ValueError('Performance roots/labels shape')
            inp,out,blocks=self.sampler.sample_blocks(roots,self.updates)
            with t.measure('request_mapping',cuda=True):
                req=self.request_from_blocks(self.arm,self.native.graph,inp,out,blocks,roots)
            with t.measure('feature_fetch',cuda=True):
                x=self.provider(req)
            if x.shape!=(len(inp),128) or x.dtype!=torch.float32 or x.device!=self.device or x.requires_grad:raise ValueError('Feature tensor contract')
            if self.updates in (0,1,2,3):
                pick=np.linspace(0,len(inp)-1,32,dtype=np.int64)
                actual=x[torch.as_tensor(pick,device=self.device)].detach().cpu().numpy()
                if not np.array_equal(actual,features(req.logical_ids[pick])):raise RuntimeError('Actual SSD/cache feature values differ from logical source')
                self.feature_checks.append(dict(passed=True,rows=32,dim=128,batch=self.updates))
            with t.measure('block_label_transfer',cuda=True):
                blocks=[b.to(self.device) for b in blocks];target=torch.from_numpy(labels.copy()).to(self.device)
            with t.measure('forward_loss',cuda=True):
                self.model.train();loss=torch.nn.functional.cross_entropy(self.model(blocks,x),target)
            with t.measure('zero_grad',cuda=True):self.opt.zero_grad(set_to_none=True)
            with t.measure('backward',cuda=True):loss.backward()
            with t.measure('runtime_check'):self.check()
            with t.measure('adam',cuda=True):self.opt.step()
            self.updates+=1
            answer=loss.detach(),len(inp)
        if t.active:t.batches+=1
        return answer
