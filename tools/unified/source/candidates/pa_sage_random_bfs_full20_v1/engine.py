"""Same PA workload in both arms; UKS cache/sampler/overlap design."""
import math
import time
import numpy as np
from .common import *

def run(arm,folder,smoke=False):
    if smoke:
        from candidates.pa_sage_random_bfs_v1.engine import run as short_run
        return short_run(arm,folder,True)
    from candidates.pa_sage_uks_style_v1.runtime import prepare_imports,loader_kwargs,arm_budget
    cls,native=prepare_imports();import torch,dgl
    from .dataset import graph_sampler,roots
    from candidates.pa_sage_uks_style_v1.backend import install_on_loader
    from candidates.pa_sage_uks_style_v1.counters import snapshot,interval
    from candidates.pa_sage_uks_style_v1.admission import admit,phase
    from candidates.uks_group_first_cpu2_v1.affinity import apply_affinity,snapshot as affinity_snapshot,validate_affinity
    from candidates.pa_sage_cache_policy_v1.training import seed_cpu,model_hash
    from candidates.uks_pa_fast_v1.monitor import snapshot as host_snapshot
    from ae.pa_sage.common import source_config
    from ae.common import check_device,check_payload
    from ae.igb.models import SAGE
    affinity=apply_affinity('gids' if arm=='gids' else 'digit_full')
    p=cfg();proof=prepared();check_device();plan=admit(arm);write(folder/'admission.json',plan)
    from candidates.pa_sage_uks_style_v1.labels import classification_targets
    labels=np.load(source_config()['label_identity']['path'],mmap_mode='r').reshape(-1)
    count=(p['train_count']+p['batch_size']-1)//p['batch_size']
    started=time.perf_counter()
    def mark(stage,**kw):write(folder/'progress.json',dict(stage=stage,arm=arm,updated_unix=time.time(),**kw))
    mark('graph_loading');graph,arrays,artifact,sampler=graph_sampler(arm,p['seed'])
    phase(folder,plan,'before_native',['native','cpu_map','degree','model','pipeline'])
    if arm=='gids':
        payload=check_payload('papers_gids');rows=payload['bytes']//512
        primary=np.arange(p['graph']['nodes'],dtype=np.int64)
        storage=np.full(rows,-1,dtype=np.int64);storage[:len(primary)]=primary
        hot=np.load(DATA/'revpr_hot.npy')
    else:
        base=ROOT/p['base_layout'];payload=read(base/'ssd_ready.json')
        require(payload['full_readback_passed'] and sha(payload['state'])==payload['state_sha256'],'DiGiT SSD state changed')
        require(sha(payload['verify_receipt'])==payload['verify_receipt_sha256'],'DiGiT SSD receipt changed')
        # Validate payload against the original feature layout, not adjacency overlay.
        from digit.artifacts import load_artifact_bundle
        from digit.ssd_payload import validate_bundle_verify_receipt
        original=load_artifact_bundle(base/'final/bundle',validation_mode='fast')
        validate_bundle_verify_receipt(payload['verify_receipt'],original,device_offset_bytes=payload['offset'])
        rows=original.manifest['feature']['num_storage_rows'];primary=artifact.arrays['node_to_primary_row'];storage=artifact.arrays['storage_to_node']
        from candidates.pa_sage_uks_style_v1.provenance import check_prepared_source
        prof=read(DATA/'profile.json');require(prof['passed'],'Incomplete independent profile')
        check_prepared_source(prof['source_sha256']);check_binding(prof['hot'])
        hot=np.load(prof['hot']['path'])
    mark('cpu_cache_install')
    loader=cls(**loader_kwargs(p,arm,rows,payload['offset']))
    phase(folder,plan,'before_cpu_cache',['cpu_map','degree','model','pipeline'])
    cache=install_on_loader(loader,native,arm_budget(p,arm),hot,primary,storage)
    phase(folder,plan,'before_model',['model','pipeline'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed_cpu(p['seed']);net=SAGE(128,p['hidden'],p['classes'],p['layers'],p['dropout']).cuda()
    kwargs=dict(p['optimizer']['kwargs']);kwargs['betas']=tuple(kwargs['betas']);opt=torch.optim.Adam(net.parameters(),**kwargs)
    initial=model_hash(net);seed_cpu(p['seed']);net.train()
    reader=None
    if arm=='digit':
        from candidates.pa_sage_uks_style_v1.model import optimize
        from candidates.pa_sage_uks_style_v1.pipeline import Reader
        phase(folder,plan,'before_pipeline',['pipeline'])
        optimize(net,opt,capture=True,batch_size=1024)
        reader=Reader(loader,1024*math.prod(1+f for f in p['fanouts']))
    from candidates.cache_epoch_v1.core import digest_array
    from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP,DIGIT_SAMPLED_GROUPS
    loader.BAM_FS.begin_useful_io_region();cold=snapshot(loader.BAM_FS)
    require(cold['gpu']['resident_pages']==0,'GPU warmed by CPU preload')
    setup_seconds=time.perf_counter()-started;epochs=[];total_rows=0;updates=0
    owners=(id(net),id(opt),id(loader),id(loader.BAM_FS),id(sampler),id(graph),id(reader))
    try:
        for epoch in range(p['epochs']):
            order_start=time.perf_counter();root=roots(epoch,arm)
            require(len(root)==p['train_count'],'Incomplete training order')
            targets=classification_targets(labels,root,p['classes']);order_seconds=time.perf_counter()-order_start
            def batches():
                for i in range(count):
                    batch=sampler.sample_blocks(graph,torch.from_numpy(root[i*1024:(i+1)*1024].copy()).cuda())
                    if arm=='gids':
                        batch[2][0].srcdata[DIGIT_STORAGE_ROW]=batch[0]
                        batch[2][0].srcdata[DIGIT_STORAGE_IS_GROUP]=torch.zeros_like(batch[0],dtype=torch.bool)
                    yield batch
            it=iter(batches());epoch_before=snapshot(loader.BAM_FS);windows=[];shapes=[];losses=[];epoch_rows=0
            for lo,hi in epoch_windows(count):
                before=snapshot(loader.BAM_FS);host_before=host_snapshot();seen=[];pending=[];group_counts=[];nr=0
                torch.cuda.synchronize();begin=time.perf_counter();iterator=reader.window(it,hi-lo) if reader else None
                for i in range(lo,hi):
                    inp,out,blocks,x=next(iterator) if iterator else loader.fetch_feature(128,it,torch.device('cuda:0'))
                    seen.append(out.detach());nr+=len(inp)
                    target=targets[i*1024:(i+1)*1024].cuda()
                    pred=net(blocks,x);loss=torch.nn.functional.cross_entropy(pred,target)
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step();pending.append(loss.detach())
                    if arm=='digit':group_counts.append(blocks[0].dstdata[DIGIT_SAMPLED_GROUPS].sum()*2)
                    shapes.append([len(inp),len(out),*[b.num_edges() for b in blocks]])
                if iterator:
                    try:next(iterator);raise RuntimeError('Pipeline crossed window')
                    except StopIteration:pass
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                after=snapshot(loader.BAM_FS);region=interval(before,after,nr)
                require(np.array_equal(torch.cat(seen).cpu().numpy(),root[lo*1024:hi*1024]),'Root order changed')
                value=torch.stack(pending).cpu().numpy();require(np.isfinite(value).all(),'Nonfinite loss')
                losses.extend(map(float,value));epoch_rows+=nr;updates+=hi-lo
                group_edges=int(torch.stack(group_counts).sum()) if group_counts else 0
                window=dict(phase='training',batch_start=lo,batch_end=hi,seconds=seconds,region=region,
                            group_sampled_edges=group_edges,outer_sampled_edges=sum(row[2] for row in shapes[lo:hi]),
                            host_before=host_before,host_after=host_snapshot())
                windows.append(window);write(folder/'current_windows.json',dict(epoch=epoch+1,windows=windows))
                mark('training',epoch=epoch+1,batch=hi,batches=count,updates=updates,total_updates=p['epochs']*count)
            try:next(it);raise RuntimeError('Extra epoch batch')
            except StopIteration:pass
            require(all(torch.isfinite(v).all().item() for v in net.parameters()),'Nonfinite parameters')
            require(owners==(id(net),id(opt),id(loader),id(loader.BAM_FS),id(sampler),id(graph),id(reader)),'Training state recreated')
            require(len(opt.state)==len(list(net.parameters())) and all(int(state['step'])==updates for state in opt.state.values()),'Optimizer update count changed')
            checkpoint=folder/('epoch%02d.pt'%(epoch+1));torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},checkpoint)
            record=dict(epoch=epoch+1,updates=count,examples=len(root),root_sha256=digest_array(root),
                        seconds=sum(w['seconds'] for w in windows),order_seconds=order_seconds,windows=windows,
                        training=interval(epoch_before,after,epoch_rows),losses=losses,shapes=shapes,persistent_state=True,
                        model_sha256=model_hash(net),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
            epochs.append(record);total_rows+=epoch_rows;write(folder/'epochs.json',epochs)
            mark('epoch_complete',epoch=epoch+1,updates=updates,seconds=record['seconds'])
        pipeline=reader.summary() if reader else dict(enabled=False)
        if reader:require(pipeline['submitted']==pipeline['collected']==updates and pipeline['outstanding']==0,'Pipeline incomplete')
        final_snapshot=snapshot(loader.BAM_FS);full=interval(cold,final_snapshot,total_rows,complete_region=True)
        final=model_hash(net);require(initial!=final,'Model did not update')
        aff=dict(initial=affinity,final=affinity_snapshot());validate_affinity(aff,'gids' if arm=='gids' else 'digit_full')
        if arm=='digit':require(sampler._cuda_call_counter==updates and net._capture_calls==p['epochs']*(p['train_count']//1024),'Sampler/capture/tail fallback missing')
        return dict(passed=True,arm=arm,smoke=False,updates=updates,warmup_batches=0,measured_batches=updates,
             measured_seconds=sum(e['seconds'] for e in epochs),setup_seconds=setup_seconds,epochs=epochs,
             training=full,complete_region=full,cache=cache,pipeline=pipeline,affinity=aff,
             root_order_policy=p['root_order_by_arm'][arm],initial_model_sha256=initial,final_model_sha256=final,
             source_sha256=verify(),protocol=p,accuracy_evaluated=False,full_epoch_training=True,raw_ssd_writes=False,
             label_source_dtype=str(labels.dtype),target_dtype=str(targets.dtype),label_range_checked=True,
             preparation_source_sha256=proof['source_sha256'],profile_source_sha256=read(DATA/'profile.json')['source_sha256'],
             measurement_scope='All 20 full epochs including cold epoch1 and every partial tail; setup/order/report/checkpoint I/O excluded')
    finally:
        if reader:reader.close()
