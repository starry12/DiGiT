"""Fixed directed traces, original CPU features; does not touch training caches."""
import pickle,random
import numpy as np
import torch
from .common import *


def model_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rng_hash():
    return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),
        torch.get_rng_state().numpy().tobytes(),torch.cuda.get_rng_state().cpu().numpy().tobytes()))).hexdigest()


class Evaluation:
    def __init__(self,p,smoke,features,labels):
        from ae.pa_sage.trace_cache import PreparedEvaluationTrace
        from ae.pa_sage.common import splits
        self.p=p;self.smoke=smoke;self.features=features;self.labels=labels
        self.calls=dict(valid=0,test=0);self.traces={};self.expected={}
        before=rng_hash();provenance=read(HERE/'provenance.json')
        for name in (('valid',) if smoke else ('valid','test')):
            desc=provenance['traces'][name]
            t=PreparedEvaluationTrace(desc['path'],desc['sha256'])
            nodes=splits(name)
            t.validate_contract(p['nodes'],p['edges'],p['fanouts'],p['batch_size'],nodes)
            t.prepare(max_bytes=8*2**30)
            self.traces[name]=t;self.expected[name]=nodes
        require(rng_hash()==before,'Trace setup changed model RNG')

    def evaluate(self,name,model):
        require(name in self.traces and (name!='test' or self.calls[name]==0),'Unexpected evaluation')
        before=model_hash(model);rng=rng_hash();training=model.training
        start=time.perf_counter();correct=0;seen=[];count=0
        model.eval()
        try:
            with torch.no_grad():
                for i,(inp,out,blocks) in enumerate(self.traces[name].iter_batches(device=None)):
                    if self.smoke and i>=2:break
                    x=torch.from_numpy(np.array(self.features[inp.numpy()],copy=True)).cuda()
                    logits=model([b.int().to('cuda:0') for b in blocks],x)
                    require(torch.isfinite(logits).all().item(),'Nonfinite evaluation logits')
                    targets=torch.from_numpy(np.array(self.labels[out.numpy()],dtype=np.int64)).cuda()
                    correct+=(logits.argmax(1)==targets).sum().item()
                    count+=1;seen.append(out.numpy().copy())
            torch.cuda.synchronize()
        finally:model.train(training)
        elapsed=time.perf_counter()-start
        actual=np.concatenate(seen);wanted=self.expected[name][:self.p['batch_size']*2] if self.smoke else self.expected[name]
        require(np.array_equal(actual,wanted),'Incomplete or reordered evaluation split')
        require(model_hash(model)==before and rng_hash()==rng,'Evaluation altered model or RNG')
        self.calls[name]+=1
        return dict(passed=True,split=name,accuracy=correct/len(actual),correct=int(correct),examples=len(actual),
            batches=count,seconds=elapsed,model_sha256=before,model_unchanged=True,rng_unchanged=True,
            trace_sha256=self.traces[name].manifest_sha256,feature_source='original CPU mmap gather; no native cache reads')
