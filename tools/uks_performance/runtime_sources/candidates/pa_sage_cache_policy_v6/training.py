"""Four-step smoke or one complete native first epoch; no evaluation path."""
import hashlib
import time
import numpy as np
from .common import read,require,progress
from candidates.pa_sage_cache_policy_v3.trace import Trace,add_batch


def run(p,arm,mode,loader,graph,bundle,sampler,binding,output,observer):
    import torch,dgl
    from digit.sampler import DIGIT_STORAGE_ROW
    from candidates.pa_sage_cache_policy_v1.training import model_hash,trace_update
    from candidates.pa_sage_cache_policy_v1.common import digest
    from candidates.pa_sage_cache_policy_v6.counters import snapshot,interval
    from candidates.pa_sage_cache_policy_v3.native_support import model,train_ids
    ids=train_ids(binding)
    labels=np.load(binding['labels'],mmap_mode='r').reshape(-1)
    features=np.load(binding['source_features'],mmap_mode='r') if mode=='smoke' else None
    original_edges=np.load(binding['original_edges'],mmap_mode='r') if mode=='smoke' else None
    net,opt,initial=model(p)
    limit=p['execution']['smoke_updates'] if mode=='smoke' else (len(ids)+p['batch_size']-1)//p['batch_size']
    require(mode=='smoke' or limit==p['execution']['formal_updates'],'Unexpected complete epoch length')
    observer.mark('model_initialized')
    loader.BAM_FS.begin_useful_io_region();begin=snapshot(loader.BAM_FS);previous=begin
    sample_trace,storage_trace=Trace(),Trace();exact_trace=hashlib.sha256()
    started=time.perf_counter()
    roots=np.random.default_rng(np.random.SeedSequence([p['seed'],0])).permutation(ids)
    root_hash=digest(roots)
    require(root_hash==read(p['orders_file'])['hashes']['s0_e0'],'Measured roots differ from the frozen main-experiment order')
    root_gpu=torch.from_numpy(roots.copy()).cuda();targets=torch.from_numpy(labels[roots].astype(np.int64)).cuda()
    size=p['batch_size'];batches=(sampler.sample_blocks(graph,root_gpu[i*size:(i+1)*size]) for i in range(limit))
    torch.cuda.synchronize();order_seconds=time.perf_counter()-started
    windows=[];shapes=[];losses=[];finite=[];total_rows=window_rows=examples=0;window_start=0
    audit_features=hashlib.sha256()
    for i in range(limit):
        inp,out,blocks,x=loader.fetch_feature(128,batches,torch.device('cuda:0'))
        require(len(out)==min(size,len(ids)-i*size),'Incomplete sampled output batch')
        finite.append((out==root_gpu[i*size:i*size+len(out)]).all())
        # Checksums run on GPU. Large sampling tensors are copied only in untimed smoke.
        add_batch(sample_trace,storage_trace,i,inp,out,blocks)
        if mode=='smoke':
            from candidates.pa_sage_direction_v2.graph import verify_sample
            verify_sample(blocks,original_edges,p['graph']['nodes'],'bidirectional')
            logical=inp.cpu().numpy();rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
            require(rows.min()>=0 and rows.max()<len(bundle.arrays['storage_to_node']) and
                    np.array_equal(bundle.arrays['storage_to_node'][rows],logical),'Sampler storage inverse differs')
            got=x.detach().cpu().numpy();want=np.array(features[logical],copy=True)
            require(np.array_equal(got.view(np.uint32),want.view(np.uint32)) and np.isfinite(got).all(),'Native feature values differ')
            audit_features.update(got.tobytes());trace_update(exact_trace,inp,out,blocks)
        pred=net(blocks,x);loss=torch.nn.functional.cross_entropy(pred,targets[i*size:i*size+len(out)])
        opt.zero_grad(set_to_none=True);loss.backward()
        checks=[torch.isfinite(pred).all(),torch.isfinite(loss)]
        checks += [torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None]
        finite.append(torch.stack(checks).all());opt.step();losses.append(loss.detach())
        shapes.append(dict(input_nodes=len(inp),output_nodes=len(out),block_edges=[b.num_edges() for b in blocks]))
        examples+=len(out);total_rows+=len(inp);window_rows+=len(inp)
        if (i+1)%p['execution']['window_updates']==0 or i+1==limit:
            torch.cuda.synchronize();current=snapshot(loader.BAM_FS)
            require(bool(torch.stack(finite).all().item()),'Nonfinite model computation or changed output roots')
            finite=[]
            windows.append(dict(start_update=window_start+1,end_update=i+1,updates=i+1-window_start,
                                region=interval(previous,current,window_rows)))
            previous=current;window_rows=0;window_start=i+1
            progress(output,'training',arm=arm,mode=mode,updates=i+1,expected_updates=limit)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    require(examples==min(len(ids),limit*size) and len(losses)==limit,'Incomplete epoch coverage')
    require(all(bool(torch.isfinite(v).all()) for v in net.parameters()),'Nonfinite final parameters')
    require(sampler._cuda_call_counter==limit,'Every batch must use native grouped sampling')
    region=interval(begin,previous,total_rows,complete_region=True)
    observer.mark('training_complete')
    values=torch.stack(losses).cpu().numpy()
    return dict(kind='native_cache_policy_short' if mode=='smoke' else 'native_cache_policy_full_epoch',
        passed=True,native=True,fixture=False,source_only=False,arm=arm,epochs=1,updates=limit,
        evaluation='disabled',finite_loss_and_gradients=True,training_seconds=seconds,
        order_seconds=order_seconds,order_excluded_seconds=seconds-order_seconds,
        initial_model_sha256=initial,final_model_sha256=model_hash(net),order_sha256=root_hash,
        sample_trace_sha256=sample_trace.digest(),storage_trace_sha256=storage_trace.digest(),
        trace_method=p['execution']['full_trace'],trace_overhead=p['execution']['trace_overhead'],
        exact_smoke_trace_sha256=exact_trace.hexdigest() if mode=='smoke' else None,
        exact_smoke_features_sha256=audit_features.hexdigest() if mode=='smoke' else None,
        feature_bit_exact=mode=='smoke',sample_edges_verified=mode=='smoke',
        windows=windows,region=region,shapes=shapes,examples=examples,losses=values.tolist(),
        feature_seconds=loader.feature_time,sampling_seconds=loader.sample_time,
        cpu_rows=p['arms'][arm]['cpu_rows'],gpu_feature_cache_bytes=p['arms'][arm]['gpu_feature_cache_bytes'],
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        optimizer_effective={k:v for k,v in opt.param_groups[0].items() if k!='params'})
