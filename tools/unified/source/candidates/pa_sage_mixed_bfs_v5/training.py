"""Continuous model/Adam/native-cache training with epoch boundaries outside timing."""
import math,resource
import numpy as np
import torch
from .common import *
from .order import prepare_orders,learning_rate
from .evaluation import Evaluation,model_hash
from candidates.pa_sage_cpu_affinity_v1 import affinity as cpu_affinity
from candidates.pa_sage_mixed_group_v1.worker import snapshot
from candidates.pa_sage_512b_overlap_v1.loader import batches as scheduled_batches
from .sampling_audit import verify_digit_blocks
from candidates.pa_sage_legacy_align_v1.graph import verify_blocks


def train_and_evaluate(folder,smoke,variant,p,graph,arrays,bundle,sampler,loader,model,
        optimizer,train,labels,features,binding,initial,cpu_cache,payload_extent,setup_start):
    from GIDS import GIDS_DGLDataLoader,get_sample_record,get_fetch_feature_record
    gids=variant=='gids_ae';active_binary=binary_for(variant)
    require(labels.dtype==torch.uint8 and labels.numel()==p['nodes'],'Wrong compact label storage')
    source=sha(HERE/'manifest.json');order_start=time.perf_counter()
    orders=prepare_orders(train,variant,smoke,p);order_seconds=time.perf_counter()-order_start
    write(folder/'order_preparation.json',dict(seconds=order_seconds,orders=[m for _,m in orders]))
    evaluation=Evaluation(p,smoke,features,np.load(binding['labels'],mmap_mode='r').reshape(-1))
    affinity=cpu_affinity.apply('one_core',p['bound_logical_cpu'])
    setup_seconds=time.perf_counter()-setup_start
    identities=tuple(id(x) for x in (graph,sampler,loader,model,optimizer,loader.BAM_FS))
    epochs=[]
    for epoch,(roots,order_meta) in enumerate(orders):
        require(tuple(id(x) for x in (graph,sampler,loader,model,optimizer,loader.BAM_FS))==identities,'Objects replaced')
        rate=learning_rate(p,epoch)
        for group in optimizer.param_groups:group['lr']=rate
        model.train();epoch_initial=model_hash(model)
        loader.begin_epoch();loader.reset_device_io_stats()
        before=snapshot(loader);dense_before=None if gids else model.dense_summary()
        sampler_before=0 if gids else sampler._cuda_call_counter
        samples0=get_sample_record();fetch0=get_fetch_feature_record()
        losses=[];outputs=[];shapes=[];audits=[];dtypes=set()
        progress(folder,'training',variant=variant,epoch=epoch+1,epochs=len(orders),learning_rate=rate)
        torch.cuda.synchronize();start=time.perf_counter()
        dataloader=GIDS_DGLDataLoader(graph,torch.from_numpy(roots.copy()),sampler,p['batch_size'],128,
            loader,shuffle=False,drop_last=False,num_workers=0,use_alternate_streams=False)
        try:
            for i,(inp,seeds,blocks,x) in enumerate(scheduled_batches(dataloader,loader,'legacy' if gids else 'overlap')):
                if smoke:
                    if gids:
                        verify_blocks(blocks,arrays,p['fanouts']);sampling_audit=None
                    else:
                        sampling_audit=verify_digit_blocks(blocks,arrays,p['fanouts'],bundle,sampler.random_seed+sampler_before+i)
                    if not gids:
                        rows=blocks[0].srcdata['digit_storage_row'].cpu().numpy()
                        require(np.array_equal(bundle.arrays['storage_to_node'][rows],inp.cpu().numpy()),'Storage inverse differs')
                    require(np.array_equal(x.cpu().numpy().view('u4'),np.asarray(features[inp.cpu().numpy()]).view('u4')),'Feature bits differ')
                blocks=[b.int().to(x.device) for b in blocks];dtypes.update(str(b.idtype) for b in blocks)
                logits=model(blocks,x);loss=torch.nn.functional.cross_entropy(logits,labels[seeds].long())
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                losses.append(loss.detach());outputs.append(seeds.detach())
                shapes.append(dict(input_nodes=len(inp),output_nodes=len(seeds),block_edges=[b.num_edges() for b in blocks]))
                if smoke:
                    require(torch.isfinite(logits).all().item(),'Nonfinite smoke logits')
                    audits.append(dict(batch=i,bit_exact=True,sampled_edges_exact=True,sampling=sampling_audit))
                if (i+1)%100==0:progress(folder,'training',variant=variant,epoch=epoch+1,updates=i+1,learning_rate=rate)
        except BaseException as error:
            write(folder/'runtime_failure.json',dict(epoch=epoch+1,updates=len(shapes),error=str(error),last_read=loader.last_read,
                sampling_details=getattr(error,'details',None)))
            raise
        finally:loader.drain()
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        after=snapshot(loader);samples1=get_sample_record();fetch1=get_fetch_feature_record()
        values=torch.stack(losses).cpu().numpy();seen=torch.cat(outputs).cpu().numpy()
        require(np.isfinite(values).all() and dtypes=={'torch.int32'},'Invalid training values/block type')
        gpu=dict(after['gpu'])
        for k in ('requests','hits','inserts','evictions','physical_inserts','partial_fills'):
            if k in gpu:gpu[k]-=before['gpu'].get(k,0)
        gpu['hit_rate']=gpu['hits']/gpu['requests'] if gpu['requests'] else 0.
        dense=None if gids else model.dense_summary()
        if dense is not None:
            for k in ('graph_forwards','eager_forwards'):dense[k]-=dense_before[k]
        io=after['device']
        # Original GIDS binary has fixed 512-B commands and no size histogram.
        # Derive its two counters from device accounting and retain that origin.
        pair_counts=([io['completed_commands']-io['replay_commands'],0,0,io['replay_commands'],0,0,0,0,0]
                     if gids else loader.BAM_FS.get_pair_counts())
        feature_access={k:after['feature'][k]-before['feature'][k] for k in before['feature']}
        affinity['at_training_end']=cpu_affinity.thread_masks()
        r=dict(passed=True,smoke=smoke,epoch=epoch+1,learning_rate=rate,variant=variant,read_mode='single' if gids else 'mixed',
            order=order_meta,root_order_exact=np.array_equal(seen,roots),root_coverage_exact=np.array_equal(np.sort(seen),np.sort(roots)),
            root_order_sha256=hashlib.sha256(seen.tobytes()).hexdigest(),epoch_initial_model_sha256=epoch_initial,
            initial_model_sha256=initial,final_model_sha256=model_hash(model),finite_parameters=all(torch.isfinite(v).all().item() for v in model.parameters()),
            examples=len(seen),updates=len(shapes),tail_size=shapes[-1]['output_nodes'],losses=values.tolist(),shapes=shapes,
            pipeline=loader.pipeline.summary(),sample_count=samples1[0]-samples0[0],fetch_count=fetch1[0]-fetch0[0],
            feature_access=feature_access,logical_feature_requests=sum(s['input_nodes'] for s in shapes),gpu_cache=gpu,
            device_io=io,pair_counts=pair_counts,
            histogram_origin='fixed_512b_backend_and_device_accounting' if gids else 'device_completion_histogram',
            timing=dict(e2e_seconds=seconds,ssd_active_seconds=after['device']['active_ns']/1e9,setup_seconds=setup_seconds if epoch==0 else 0.),
            cpu_cache=cpu_cache,cpu_cache_rows=p['cpu_rows'],page_bytes=512,window_buffer=True,accumulator=True,mixed_io=not gids,
            arm='gids' if gids else 'digit',mean_mode='legacy',mean_normalize_api=0 if gids else 1,normalization_binary_sha256=None if gids else sha(NORMALIZATION_BINARY),
            group_mode='standard_uniform' if gids else 'incremental',incremental_group_api=0 if gids else 1,
            sampler_binary_sha256=None if gids else sha(SAMPLER_BINARY),dense_mode='eager' if gids else 'graph',dense_graph=dense,
            io_mode='legacy' if gids else 'overlap',eid_mode='original',async_backend=dict(outstanding=0) if gids else loader.async_summary(),changes=dict(frontier=not gids,functional=not gids),
            model_implementation='standard_sage' if gids else 'last_dense_cuda_graph',frontier_implementation='dgl_uniform' if gids else 'known_extent_coo',additional_sampling_profile=False,
            diagnostic_trace=False,fifo_window_512b_api=1,payload_extent=payload_extent,affinity=dict(affinity),block_index_bits=32,
            fifo_window_policy='ring ticket with protected/busy victim skipping at 512 B',
            continuous_cpu_frequency_observation=False,native_grouped_batches=0 if gids else sampler._cuda_call_counter-sampler_before,
            smoke_checks=dict(features_bit_exact=bool(smoke and audits),sampled_edges_exact=bool(smoke and audits),storage_inverse_exact=bool(not gids and smoke and audits)),
            audits=audits,protocol=p,candidate_sha256=source,binary_sha256=sha(active_binary),input_binding_sha256=sha(OUT/'inputs.json'),
            raw_ssd_writes=False,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        validate_epoch(r)
        progress(folder,'validation',variant=variant,epoch=epoch+1)
        # Release the drained training batch before the disjoint evaluation phase.
        del dataloader,blocks,x,logits,loss,inp,seeds,losses,outputs
        r['validation']=evaluation.evaluate('valid',model)
        require(snapshot(loader)==after,'Evaluation touched native training cache/statistics')
        epochs.append(r);write(folder/('epoch_%02d.json'%(epoch+1)),r)
        progress(folder,'epoch_complete',variant=variant,epoch=epoch+1,training_seconds=seconds,validation_accuracy=r['validation']['accuracy'])
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},folder/'final_model.pt')
    test=None
    if not smoke:
        write(folder/'test_invocation.json',dict(epoch=len(epochs),checkpoint_sha256=sha(folder/'final_model.pt'),model_sha256=model_hash(model)))
        progress(folder,'test',variant=variant,epoch=len(epochs));test=evaluation.evaluate('test',model)
    return dict(passed=True,smoke=smoke,variant=variant,epochs=epochs,test=test,
        label_storage=dict(dtype='uint8',bytes=labels.numel()),
        protocol=p,candidate_sha256=source,binary_sha256=sha(active_binary),initial_model_sha256=initial,
        final_model_sha256=model_hash(model),evaluation_calls=evaluation.calls,objects_reused=True,
        training_seconds=sum(e['timing']['e2e_seconds'] for e in epochs),setup_seconds=setup_seconds,
        order_preparation_seconds=order_seconds,validation_seconds=sum(e['validation']['seconds'] for e in epochs),
        cpu_cache=cpu_cache,checkpoint_sha256=sha(folder/'final_model.pt'))
