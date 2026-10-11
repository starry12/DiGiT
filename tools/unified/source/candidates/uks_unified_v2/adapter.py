"""Retain UKS model, optimizer and native caches for every common-loop epoch."""
import time
import numpy as np
from candidates.unified_training_v2.loop import Adapter
from candidates.unified_training_v2.order import roots_digest
from candidates.unified_training_v2.accounting import validate_counts

def counts(raw,groups,edges,arm):
    if not raw['reconciled']:raise ValueError('Unreconciled UKS interval')
    s=raw['serving'];d=raw['device'];z=raw['request_sizes']
    c=dict(logical_rows=s['logical_requests'],cpu_rows=s['cpu_served_rows'],gpu_rows=s['gpu_hit_rows'],ssd_rows=s['ssd_served_rows'],
        primary_single_commands=z['primary_1k'],primary_pair_commands=z['primary_2k'],primary_bytes=d['primary_bytes'],
        replay_commands=d['replay_commands'],replay_bytes=d['replay_bytes'],completed_commands=d['completed_commands'],
        completed_bytes=d['completed_bytes'],sampled_group_edges=groups,sampled_outer_edges=edges)
    validate_counts(c,1024,arm);return c

class UKSAdapter(Adapter):
    def __init__(self,resources):self.r=resources;self.reader=None;self.updates=0;self.checked=0;self.full_batches=0
    def initialize(self,config,arm):
        from candidates.pa_sage_cache_policy_v1.training import model_hash
        if config['dataset']!='UKS':raise ValueError('UKS only')
        self.arm=arm;r=self.r;self.config=config
        if any(g['weight_decay']!=config['data']['weight_decay'] or g['lr']!=config['lr'] for g in r.optimizer.param_groups):
            raise ValueError('UKS actual optimizer differs from unified config')
        self.device=next(r.model.parameters()).device
        if arm=='digit':
            from candidates.uks_pa_fast_v1.model import optimize
            from candidates.uks_pa_fast_v1.pipeline import Reader
            optimize(r.model,r.optimizer,capture=self.device.type=='cuda',batch_size=config['batch_size'])
            if self.device.type=='cuda':
                import torch
                if torch.cuda.mem_get_info()[0]<=2*2**30:raise RuntimeError('UKS prefetch needs 2 GiB additional headroom')
                self.reader=Reader(r.loader,405504)
        r.loader.BAM_FS.begin_useful_io_region()
        return dict(r.receipt,initial_model_sha256=model_hash(r.model))
    def run_window(self,batches,epoch,phase):
        import torch
        from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP,DIGIT_SAMPLED_GROUPS
        r=self.r;r.check();r.model.train();before=r.snapshot();actual=[];losses=[];nr=edges=0;groups=[]
        def sampled():
            for roots in batches:
                root=torch.from_numpy(np.array(roots,copy=True)).to(self.device)
                inp,out,blocks=r.sampler.sample_blocks(r.graph,root)
                if self.arm=='gids':
                    blocks[0].srcdata[DIGIT_STORAGE_ROW]=inp
                    blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]=torch.zeros_like(inp,dtype=torch.bool)
                yield inp,out,blocks
        sync=torch.cuda.synchronize if self.device.type=='cuda' else lambda:None
        iterator=iter(sampled());sync();started=time.perf_counter()
        pending=self.reader.window(iterator,len(batches)) if self.reader else None
        for roots in batches:
            inp,out,blocks,x=next(pending) if pending else r.loader.fetch_feature(256,iterator,self.device)
            actual.append(out.detach());nr+=len(inp);edges+=blocks[0].num_edges()
            if self.arm=='digit':groups.append(2*blocks[0].dstdata[DIGIT_SAMPLED_GROUPS].sum())
            if self.checked<4:
                pick=torch.linspace(0,len(inp)-1,min(32,len(inp)),device=inp.device).long()
                if not np.array_equal(x[pick].detach().cpu().numpy(),np.asarray(r.features[inp[pick].cpu().numpy()])):
                    raise RuntimeError('UKS feature/reference mismatch')
                self.checked+=1
            target=torch.from_numpy(np.array(r.labels[roots],dtype=np.int64,copy=True)).to(self.device)
            loss=torch.nn.functional.cross_entropy(r.model(blocks,x),target)
            r.optimizer.zero_grad(set_to_none=True);loss.backward();r.optimizer.step();losses.append(loss.detach());self.updates+=1
            self.full_batches+=int(len(roots)==self.config['batch_size'])
        if pending:
            try:next(pending);raise RuntimeError('Prefetch crossed window')
            except StopIteration:pass
        sync();seconds=time.perf_counter()-started
        delivered=torch.cat(actual).cpu().numpy()
        if not np.array_equal(delivered,np.concatenate(batches)):raise RuntimeError('Delivered roots changed')
        raw=r.interval(before,r.snapshot(),nr,complete_region=False)
        return dict(seconds=seconds,updates=len(batches),roots=len(delivered),roots_sha256=roots_digest([delivered]),
            counters=counts(raw,int(torch.stack(groups).sum()) if groups else 0,edges,self.arm),
            finite=bool(torch.isfinite(torch.stack(losses)).all()),outstanding_io=int(self.reader is not None and self.reader.live is not None),outstanding_prefetch=0)
    def epoch_end(self,epoch):
        import torch
        p=self.r.output/('epoch_%02d.pt'%(epoch+1))
        torch.save(dict(model=self.r.model.state_dict(),optimizer=self.r.optimizer.state_dict(),updates=self.updates),p)
        return dict(evaluated=False,checkpoint=str(p),scope='synthetic_labels_no_accuracy_claim')
    def finish(self):
        import torch
        from candidates.pa_sage_cache_policy_v1.training import model_hash
        if self.checked!=4:raise RuntimeError('Missing actual feature checks')
        if self.arm=='digit' and self.r.sampler._cuda_call_counter!=self.updates:raise RuntimeError('Sampler epoch reset')
        if self.arm=='digit' and self.device.type=='cuda' and self.r.model._capture_calls!=self.full_batches:
            raise RuntimeError('UKS capture/tail fallback count mismatch')
        return dict(updates=self.updates,final_model_sha256=model_hash(self.r.model),finite=all(bool(torch.isfinite(p).all()) for p in self.r.model.parameters()),
            outstanding_io=int(self.reader is not None and self.reader.live is not None),outstanding_prefetch=0,feature_checks=self.checked)
    def close(self):
        if self.reader:self.reader.close()
        # Native allocations remain owned until this fresh worker exits.
        # Parent verifies normal exit and GPU release before accepting reports.
