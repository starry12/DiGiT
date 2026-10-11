"""Owned native sampling -> real feature requests -> CUDA mean-SAGE/Adam."""
import copy,time
import numpy as np
import torch,dgl
from candidates.ukl_training_adapter_v12.adapter import prepare_model,request_from_blocks,model_hash
from . import sampling as S
from candidates.ukl_native_sampling_v10r4 import fork_guard as H
from .backend import Provider


def prepare_gpu_model(seed=0):
    if H._ACTIVE:raise RuntimeError('GPU model warmup must precede graph ownership')
    model,_=prepare_model(seed) # CPU lazy initialization precedes CUDA.
    torch.cuda.set_device(0);torch.cuda.init();torch.cuda.manual_seed(seed)
    scratch=copy.deepcopy(model).cuda();opt=torch.optim.Adam(scratch.parameters(),lr=.001,weight_decay=.001)
    b=dgl.create_block((torch.tensor([0]),torch.tensor([0])),num_src_nodes=1,num_dst_nodes=1).to('cuda:0')
    x=torch.ones((1,128),device='cuda:0');target=torch.zeros(1,dtype=torch.int64,device='cuda:0')
    loss=torch.nn.functional.cross_entropy(scratch([b,b,b],x),target);loss.backward();opt.step();torch.cuda.synchronize()
    del scratch,opt,b,x,target,loss
    model=model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
    torch.cuda.manual_seed(seed);torch.cuda.empty_cache()
    return model,optimizer


class Engine:
    def __init__(self,native,arm,model,optimizer,provider,check=lambda:None,fixture=False):
        Native, Sampler = S.select(arm)
        if type(native)is not Native or arm not in ('gids','digit') or not H._ACTIVE:raise ValueError('Exact owned native sampler required')
        H.assert_dontfork(native.graph.arena);self.device=next(model.parameters()).device
        if fixture:
            if native.gpu or native.graph.nodes>4096 or self.device.type!='cpu':raise ValueError('CPU tiny fixture scope')
        elif not native.gpu or self.device.type!='cuda' or type(provider)is not Provider:raise RuntimeError('Production requires admitted GPU sampler and real provider')
        if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(p) for p in model.parameters()}:raise ValueError('Optimizer ownership')
        self.native,self.arm,self.model,self.opt,self.provider,self.check,self.fixture=native,arm,model,optimizer,provider,check,fixture
        self.sampler=Sampler(native,grouped=arm=='digit',seed=0);self.updates=0
    def step(self,roots,labels):
        self.check();roots=np.asarray(roots);labels=np.asarray(labels)
        if roots.ndim!=1 or labels.shape!=roots.shape or labels.dtype!=np.int64 or np.any(labels<0) or np.any(labels>=19):raise ValueError('Window labels')
        inp,out,blocks=self.sampler.sample_blocks(roots,self.updates)
        request=request_from_blocks(self.arm,self.native.graph,inp,out,blocks,roots)
        x=self.provider(request)
        if x.shape!=(len(inp),128) or x.dtype!=torch.float32 or x.device!=self.device or x.requires_grad:raise ValueError('Feature tensor contract')
        blocks=[b.to(self.device) for b in blocks];target=torch.from_numpy(labels.copy()).to(self.device)
        self.model.train();pred=self.model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,target)
        self.opt.zero_grad(set_to_none=True);loss.backward();self.check();self.opt.step();self.updates+=1
        return loss.detach(),len(inp)
    def window(self,roots,labels,warmup=20,measured=300,event=lambda **kw:None):
        roots=np.asarray(roots);labels=np.asarray(labels)
        if roots.ndim!=2 or labels.shape!=roots.shape or roots.dtype!=np.int64 or labels.dtype!=np.int64 or len(roots)!=warmup+measured or warmup<0 or measured<=0 or self.updates:raise ValueError('Complete fresh window required')
        if self.fixture:
            if roots.size>4096 or len(roots)>16:raise ValueError('Tiny fixture window')
        elif (warmup,measured,roots.shape[1])!=(20,300,1024) and (warmup,measured,roots.shape[1])!=(0,4,1024):raise ValueError('Only 4-batch smoke or 20+300 protocol')
        sync=(lambda:None) if self.fixture else torch.cuda.synchronize
        snapshot=(lambda:None) if self.fixture else self.provider.snapshot
        before=snapshot();initial=model_hash(self.model);pending=[];rows=0;measured_rows=0;measure_before=None;started=None
        sync();warm_start=time.perf_counter()
        for i in range(len(roots)):
            if i==warmup:
                sync();warm_seconds=time.perf_counter()-warm_start;measure_before=snapshot();started=time.perf_counter()
            loss,count=self.step(roots[i],labels[i]);pending.append(loss);rows+=count
            if i>=warmup:measured_rows+=count
            event(stage='training' if i>=warmup else 'warmup',updates=self.updates,total=len(roots))
        sync();seconds=time.perf_counter()-started;after=snapshot()
        if not torch.isfinite(torch.stack(pending)).all().item() or any(not torch.isfinite(p).all().item() for p in self.model.parameters()):raise RuntimeError('Nonfinite training')
        final=model_hash(self.model)
        if final==initial:raise RuntimeError('No model update')
        counts={}
        if not self.fixture:
            from .counters import interval
            counts=dict(whole=interval(before,after,rows,True),measured=interval(measure_before,after,measured_rows,False))
        return dict(passed=True,updates=self.updates,warmup_batches=warmup,measured_batches=measured,seconds=seconds,warmup_seconds=warm_seconds,
          initial_model_sha256=initial,final_model_sha256=final,finite=True,counters=counts,accuracy_evaluated=False)
