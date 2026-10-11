"""Persistent UKL/CL training, device requests, bounded prefetch and tail fallback."""
import time
import numpy as np
from candidates.unified_training_v2.loop import Adapter as CommonAdapter
from candidates.unified_training_v2.order import roots_digest
from candidates.unified_results_v2.report import raw_counts

class Adapter(CommonAdapter):
    def __init__(self,resources):
        self.r=resources;self.updates=self.full_batches=self.groups=self.edges=self.rows=self.measured_rows=0
        self.feature_checks=[];self.window_records=[];self.reader=None;self.measured_before=None
    def initialize(self,config,arm):
        from candidates.ukl_training_adapter_v12.adapter import model_hash
        if config['dataset']!=self.dataset or config['data']['weight_decay']!=0:raise ValueError('Wrong unified configuration')
        self.config=config;self.arm=arm;r=self.r;self.device=next(r.model.parameters()).device
        if r.receipt['gpu_resident_rows_start']!=0:raise ValueError('Cold GPU cache required')
        if any(g['weight_decay']!=0 or g['lr']!=config['lr'] for g in r.optimizer.param_groups):raise ValueError('Actual optimizer differs from config')
        if arm=='digit':
            from candidates.uks_pa_fast_v1.model import optimize
            optimize(r.model,r.optimizer,capture=self.device.type=='cuda',batch_size=config['batch_size'])
            if self.device.type=='cuda':
                import torch
                if torch.cuda.mem_get_info()[0]<512*2**20:raise RuntimeError('Missing prefetch headroom')
                from .pipeline import Reader
                self.reader=Reader(r.provider)
        self.before=r.snapshot();self.initial=model_hash(r.model)
        self.owners=(id(r.model),id(r.optimizer),id(r.sampler),id(r.provider))
        return dict(r.receipt,initial_model_sha256=self.initial)
    def run_window(self,batches,epoch,phase):
        import torch
        from candidates.ukl_runtime_prepare_v11r1.data import labels,features
        r=self.r;r.check();r.model.train();before=r.snapshot();actual=[];losses=[];group_counts=[];rows=edges=0;base=self.updates
        if phase=='training' and self.measured_before is None:self.measured_before=before
        def sampled():
            for i,roots in enumerate(batches):
                inp,out,blocks=r.sampler.sample_blocks(roots,base+i)
                g=r.sampler.last_group_counts['grouped_sampled_edges'] if self.arm=='digit' else 0
                yield inp,out,blocks,roots,g
        def synchronous():
            for batch in sampled():
                if self.device.type=='cuda':x=r.provider.fetch_device(batch)
                else:x=r.provider(r.request(self.arm,r.native.graph,*batch[:3],batch[3]))
                yield batch,x
        sync=torch.cuda.synchronize if self.device.type=='cuda' else lambda:None
        sync();start=time.perf_counter()
        stream=self.reader.window(sampled()) if self.reader else synchronous()
        for batch,x in stream:
            inp,out,blocks,roots,g=batch;actual.append(out.detach());rows+=len(inp);edges+=blocks[0].num_edges();group_counts.append(g)
            if x.shape!=(len(inp),128) or x.dtype!=torch.float32 or x.device!=self.device:raise ValueError('Feature contract')
            if self.updates<4:
                pick=torch.linspace(0,len(inp)-1,min(32,len(inp)),device=inp.device).long()
                ids=inp[pick].cpu().numpy()
                if not np.array_equal(x[pick.to(x.device)].detach().cpu().numpy(),features(ids)):raise RuntimeError('Actual features differ from logical reference')
                self.feature_checks.append(dict(passed=True,rows=len(pick),dim=128,batch=self.updates))
            target=torch.from_numpy(labels(roots)).to(self.device)
            loss=torch.nn.functional.cross_entropy(r.model(blocks,x),target)
            r.optimizer.zero_grad(set_to_none=True);loss.backward();r.optimizer.step()
            losses.append(loss.detach());self.updates+=1;self.full_batches+=int(len(roots)==self.config['batch_size'])
        sync();seconds=time.perf_counter()-start
        delivered=torch.cat(actual).cpu().numpy().astype(np.int64,copy=False)
        if not np.array_equal(delivered,np.concatenate(batches)):raise ValueError('Delivered roots changed')
        groups=int(sum(group_counts));after=r.snapshot();raw=r.interval(before,after,rows,False)
        self.rows+=rows;self.groups+=groups;self.edges+=edges
        if phase=='training':self.measured_rows+=rows
        self.window_records.append(dict(epoch=epoch+1,phase=phase,raw=raw,groups=groups,edges=edges,rows=rows))
        r.check()
        return dict(seconds=seconds,updates=len(batches),roots=len(delivered),roots_sha256=roots_digest([delivered]),
            counters=raw_counts(raw,groups,edges,self.arm),finite=bool(torch.isfinite(torch.stack(losses)).all()),
            outstanding_io=int(self.reader is not None and self.reader.live is not None),outstanding_prefetch=0)
    def epoch_end(self,epoch):
        import torch
        r=self.r
        if self.owners!=(id(r.model),id(r.optimizer),id(r.sampler),id(r.provider)):raise ValueError('Training state recreated')
        p=r.output/('epoch_%02d.pt'%(epoch+1))
        torch.save(dict(model=r.model.state_dict(),optimizer=r.optimizer.state_dict(),epoch=epoch+1,updates=self.updates),p)
        return dict(evaluated=False,checkpoint=str(p),scope='training_only_no_accuracy_claim')
    def finish(self):
        import torch
        from candidates.ukl_training_adapter_v12.adapter import model_hash
        if self.arm=='digit' and self.device.type=='cuda' and self.r.model._capture_calls!=self.full_batches:raise RuntimeError('Capture/tail fallback count mismatch')
        if self.reader and (self.reader.submitted!=self.reader.collected or self.reader.live is not None):raise RuntimeError('Prefetch not drained')
        after=self.r.snapshot();self.whole=self.r.interval(self.before,after,self.rows,True)
        self.measured=self.r.interval(self.measured_before,after,self.measured_rows,False)
        return dict(updates=self.updates,final_model_sha256=model_hash(self.r.model),
            finite=all(bool(torch.isfinite(p).all()) for p in self.r.model.parameters()),outstanding_io=0,outstanding_prefetch=0,
            pipeline=self.reader.summary() if self.reader else dict(enabled=False))
    def close(self):
        if self.reader:self.reader.close()
        if getattr(self,'device',None) is not None and self.device.type=='cuda':
            import torch
            torch.cuda.synchronize()
        self.r=None
