"""Persistent IG model/cache adapter; the caller owns admission and teardown.

Resources are constructed only in initialize, after the existing guarded worker
has been admitted. Each process owns one arm; windows and epochs retain state.
"""
import hashlib,time
import numpy as np
from candidates.unified_training_v2.loop import Adapter
from candidates.unified_training_v2.config import validate
from candidates.unified_training_v2.order import roots_digest
from .accounting import counters

def model_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class IGAdapter(Adapter):
    def __init__(self,factory):self.factory=factory;self.r=None;self.updates=0;self.reader=None;self.full_batches=0
    def initialize(self,config,arm):
        validate(config)
        if config['dataset']!='IG':raise ValueError('IG adapter only')
        self.config=config;self.arm=arm;self.r=self.factory(config,arm)
        r=self.r
        if any(g['weight_decay']!=config['data']['weight_decay'] or g['lr']!=config['lr'] for g in r.optimizer.param_groups):
            raise ValueError('IG actual optimizer differs from unified config')
        if r.receipt['evidence_scope']!='native' or r.receipt['gpu_resident_rows_start']!=0:
            raise ValueError('Fresh native cache required')
        if arm=='digit':
            from candidates.uks_pa_fast_v1.model import optimize
            from .pipeline import Reader
            optimize(r.model,r.optimizer,capture=True,batch_size=config['batch_size'])
            self.reader=Reader(r.features)
        initial=dict(r.receipt,initial_model_sha256=model_hash(r.model))
        self.initial=initial;return initial
    def run_window(self,batches,epoch,phase):
        import torch,dgl
        from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP,DIGIT_SAMPLED_GROUPS
        r=self.r;r.check();r.model.train();r.features.store.begin_useful_io_region()
        before=r.features.begin();losses=[];group_counts=[];outer=rows_seen=0
        delivered=[];root_checks=[]
        def sampled():
            nonlocal outer,rows_seen
            for roots in batches:
                if roots.dtype!=np.int64 or not 0<len(roots)<=self.config['batch_size']:raise ValueError('Root batch')
                root=torch.from_numpy(np.array(roots,copy=True)).cuda()
                inp,out,blocks=r.sampler.sample_blocks(r.graph,root)
                # Check actual roots, not a hash of the caller's requested plan.
                if out.shape!=root.shape or out.dtype!=root.dtype:raise ValueError('IG sampler changed root extent')
                root_checks.append((out==root).all())
                delivered.append(roots.copy());rows_seen+=len(inp);outer+=blocks[0].num_edges()
                if self.arm=='digit':
                    physical=blocks[0].srcdata[DIGIT_STORAGE_ROW]
                    flags=blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]
                    group_counts.append(blocks[0].dstdata[DIGIT_SAMPLED_GROUPS].sum())
                else:physical=inp;flags=torch.zeros_like(inp,dtype=torch.bool)
                labels=np.asarray(r.labels[roots])
                if not np.isfinite(labels).all() or not np.all(labels==labels.astype('int64')) or labels.min()<0 or labels.max()>=19:
                    raise ValueError('Invalid IG labels')
                targets=torch.from_numpy(labels.astype('int64')).cuda()
                yield inp,out,blocks,physical,flags,targets
        def synchronous():
            for batch in sampled():yield batch,r.features.fetch_device(batch[3],batch[4])
        torch.cuda.synchronize();start=time.perf_counter()
        stream=self.reader.window(sampled()) if self.reader is not None else synchronous()
        for batch,x in stream:
            r.audit(batch[0],batch[2],x)
            pred=r.model(batch[2],x);loss=torch.nn.functional.cross_entropy(pred,batch[5])
            r.optimizer.zero_grad(set_to_none=True);loss.backward();r.optimizer.step()
            losses.append(loss.detach());self.updates+=1;self.full_batches+=int(len(batch[1])==self.config['batch_size'])
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if not bool(torch.stack(root_checks).all()):raise ValueError('IG sampler changed root order')
        if self.arm=='digit':r.sampler.check_window()
        finite=bool(torch.isfinite(torch.stack(losses)).all().item())
        groups=2*int(torch.stack(group_counts).sum()) if group_counts else 0
        raw=r.features.finish(before,rows_seen)
        r.window_evidence(epoch,phase,raw)
        r.check()
        return dict(seconds=elapsed,updates=len(losses),roots=sum(map(len,delivered)),
                    roots_sha256=roots_digest(delivered),counters=counters(raw,groups,outer,self.arm),
                    finite=finite,outstanding_io=int(self.reader is not None and self.reader.live is not None),
                    outstanding_prefetch=0)
    def epoch_end(self,epoch):
        import torch
        # Export only; no evaluation is implied by training with real labels.
        p=self.r.output/('epoch_%02d.pt'%(epoch+1))
        torch.save(dict(model=self.r.model.state_dict(),optimizer=self.r.optimizer.state_dict(),
                        epoch=epoch+1,updates=self.updates),p)
        return dict(evaluated=False,checkpoint=str(p),scope='training_only_no_accuracy_claim')
    def finish(self):
        import torch
        self.r.features.final_checks()
        if self.arm=='digit' and self.r.model._capture_calls!=self.full_batches:raise RuntimeError('IG capture/tail fallback count mismatch')
        return dict(updates=self.updates,final_model_sha256=model_hash(self.r.model),
                    finite=all(bool(torch.isfinite(v).all()) for v in self.r.model.parameters()),
                    outstanding_io=int(self.reader is not None and self.reader.live is not None),outstanding_prefetch=0)
    def close(self):
        if self.reader is not None:self.reader.close()
        if self.r is not None:self.r.close()
