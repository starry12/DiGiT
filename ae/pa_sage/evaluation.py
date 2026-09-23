"""Formal evaluation: prepare CPU templates once; exactly one pass per call."""
import hashlib
import pickle
import time
from ae.pa_sage.common import Path, read, write, require
from ae.pa_sage.trace_cache import PreparedEvaluationTrace, CACHE_BUDGET_BYTES
import runner as r
import torch


def rng_fingerprint():
    state=(r.random.getstate(),r.np.random.get_state(),torch.get_rng_state().numpy().tobytes(),
           [v.cpu().numpy().tobytes() for v in torch.cuda.get_rng_state_all()],r.dgl_rng())
    return hashlib.sha256(pickle.dumps(state,protocol=4)).hexdigest()


class EvaluationSession:
    def __init__(self,data,output,prior,protocol,smoke=False):
        self.data=Path(data);self.output=Path(output);self.prior=prior;self.protocol=protocol
        self.smoke=smoke;self.traces={};self.calls={'valid':0,'test':0};self.seconds={'valid':0.,'test':0.}

    def prepare(self):
        start=time.perf_counter();before=rng_fingerprint();gpu=torch.cuda.memory_allocated()
        ready=read(self.data/'traces_ready.json')['traces'];target=self.protocol['paper_specified']
        for split in (('valid',) if self.smoke else ('valid','test')):
            trace=PreparedEvaluationTrace(self.data/(split+'_trace'),ready[split]['manifest_sha256'])
            trace.validate_contract(self.prior.N,self.prior.E,target['fanouts'],target['batch_size'],self.prior.splits(split))
            self.traces[split]=trace
        require(sum(t.budget['required_bytes'] for t in self.traces.values())+64*2**20<=CACHE_BUDGET_BYTES,'Trace and monitor exceed host reserve')
        details={split:t.prepare() for split,t in self.traces.items()}
        require(torch.cuda.memory_allocated()==gpu,'CPU trace preparation allocated CUDA data')
        require(rng_fingerprint()==before,'Trace preparation changed training RNG')
        self.preparation=dict(seconds=time.perf_counter()-start,traces=details,host_reserve_bytes=CACHE_BUDGET_BYTES,
            independent_monitor_reserve_bytes=64*2**20,persistent_cuda_bytes=0,training_rng_unchanged=True)
        write(self.output/'trace_preparation.json',self.preparation)

    def evaluate(self,split,model,loader,bundle,labels,limit=None,audit_features=None):
        require(split in self.traces,'Unprepared evaluation split')
        require(not (split=='test' and self.calls['test']),'Final test already evaluated')
        before=rng_fingerprint();training=model.training
        result=r.evaluate(model,loader,bundle,labels,self.traces[split],limit,audit_features)
        require(rng_fingerprint()==before and model.training==training,'Evaluation changed RNG or model mode')
        self.calls[split]+=1;self.seconds[split]+=result['seconds']
        return result

    def finish(self,epochs):
        require(self.calls=={'valid':epochs,'test':0 if self.smoke else 1},'Unexpected evaluation replay/count')
        return dict(mode='formal_single_pass',validation_calls=self.calls['valid'],test_calls=self.calls['test'],
            diagnostic_replays=0,cpu_diagnostic_monitor=False,training_rng_unchanged=True,
            trace_preparation_seconds=self.preparation['seconds'],preparation=self.preparation,
            validation_seconds=self.seconds['valid'],test_seconds=self.seconds['test'])
