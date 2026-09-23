"""Selected immutable bidirectional traces, one evaluation pass per call."""
import time
from training.gcn.common import cfg,ROOT,sha,require,write
from ae.pa_sage.evaluation import EvaluationSession as Base,rng_fingerprint
from ae.pa_sage.trace_cache import PreparedEvaluationTrace
from runtime.io.accounting import begin_region,validate_region
import torch
class EvaluationSession(Base):
    def prepare(self):
        p=cfg();start=time.perf_counter();before=rng_fingerprint();gpu=torch.cuda.memory_allocated();self.traces={}
        for split in (('valid',) if self.smoke else ('valid','test')):
            path=ROOT/p['validation_trace' if split=='valid' else 'test_trace']
            trace=PreparedEvaluationTrace(path,sha(path/'manifest.json'))
            trace.validate_contract(p['graph']['nodes'],p['graph']['edges'],p['fanouts'],p['batch_size'],self.prior.splits(split))
            self.traces[split]=trace
        require(sum(t.budget['required_bytes'] for t in self.traces.values())<=16*2**30,'Trace cache exceeds candidate host budget')
        details={split:t.prepare(max_bytes=8*2**30) for split,t in self.traces.items()}
        require(torch.cuda.memory_allocated()==gpu and rng_fingerprint()==before,'Trace preparation changed GPU memory/RNG')
        self.preparation=dict(seconds=time.perf_counter()-start,traces=details,persistent_cuda_bytes=0,training_rng_unchanged=True,host_reserve_bytes=16*2**30)
        write(self.output/'trace_preparation.json',self.preparation)
    def evaluate(self,split,model,loader,*args,**kwargs):
        begin_region(loader);result=super().evaluate(split,model,loader,*args,**kwargs);validate_region(result);return result
