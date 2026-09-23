"""Configuration-driven SAGE loop; measurement and evaluation follow the extracted runtime."""
import math
from runner import *
from training.sage.evaluation import EvaluationSession
from training.sage.common import cfg,HERE,source_config
from runtime.io.accounting import begin_region,validate_region
from ae.papers.graph_checks import verify_sample

def setup_model(value):
    p=cfg();seed(value);model=SAGE(128,p['hidden'],p['classes'],num_layers=p['layers'],dropout=p['dropout']).cuda()
    kwargs=dict(p['optimizer']['kwargs']);kwargs['betas']=tuple(kwargs['betas'])
    optimizer=torch.optim.Adam(model.parameters(),**kwargs)
    initial=model_hash(model);seed(value);model.train();return model,optimizer,initial

def run_training(arm,value,loader,bundle,graph,labels,degrees,output,data,protocol,smoke=False,features=None):
    S=Path(data);D=S;TEST=S/"test_trace"
    fanouts=protocol["paper_specified"]["fanouts"];batch_size=protocol["paper_specified"]["batch_size"]
    valid_count=len(PRIOR.splits('valid'));test_count=0 if smoke else len(PRIOR.splits('test'))
    total_epochs=2 if smoke else protocol["paper_specified"]["epochs"]
    group_size=protocol["paper_specified"]["group_size"]
    sampler=(DiGiTNeighborSampler(fanouts,bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=value)
             if bundle else dgl.dataloading.NeighborSampler(fanouts,replace=False))
    train_ids=PRIOR.splits('train');order_hashes=read(ROOT/cfg()['base_layout']/'orders.json')['hashes']
    evaluation=EvaluationSession(data,output.parent,PRIOR,protocol,smoke=smoke)
    evaluation.prepare()
    model,optimizer,initial=setup_model(value);initial_rng=dgl_rng();epochs=[];audits=[];metadata=None
    original_edges=np.load(source_config()['source_contract']['original_edges']['path'],mmap_mode='r') if smoke else None
    selected_reference=read(ROOT/'reference/sage_correctness.json') if arm=='gids' and not smoke else None
    metadata_start=time.perf_counter()
    if bundle:sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    torch.cuda.synchronize();metadata_setup_seconds=time.perf_counter()-metadata_start
    from training.sage.admission import estimate
    assert torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0]<=estimate()['required_bytes']
    identities=(id(graph),id(sampler),id(loader),id(model),id(optimizer));start_cache=snapshot(loader)
    for epoch in range(total_epochs):
        assert identities==(id(graph),id(sampler),id(loader),id(model),id(optimizer))
        begin_region(loader)
        begin=snapshot(loader);previous=begin;epoch_start=time.perf_counter()
        roots=np.random.default_rng(np.random.SeedSequence([value,epoch])).permutation(train_ids)
        root_hash=digest(roots);assert root_hash==order_hashes['s%d_e%d'%(value,epoch)]
        root_gpu=torch.from_numpy(roots.copy()).cuda();ys=torch.from_numpy(labels[roots].astype(np.int64)).cuda()
        limit=2 if smoke else (len(train_ids)+batch_size-1)//batch_size
        def batches():
            for i in range(limit):yield sampler.sample_blocks(graph,root_gpu[i*batch_size:(i+1)*batch_size])
        iterator=batches();torch.cuda.synchronize();order_seconds=time.perf_counter()-epoch_start
        window_start=time.perf_counter();losses=[];coverage=[];shapes=[];windows=[];window_rows=0;total_rows=0
        for i in range(limit):
            inp,out,blocks,x=loader.fetch_feature(128,iterator,torch.device('cuda:0'))
            if smoke:verify_sample(blocks,original_edges,cfg()["graph"]["nodes"],"bidirectional")
            record=batch_audit(inp,out,blocks,x,features,bundle) if smoke else None
            if smoke:assert np.array_equal(out.cpu().numpy(),roots[i*batch_size:(i+1)*batch_size])
            pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,ys[i*batch_size:i*batch_size+len(out)])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if smoke:assert torch.isfinite(pred).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();losses.append(loss.detach())
            target=degrees[blocks[0].dstdata[dgl.NID]].to(torch.int64).clamp(max=fanouts[0]);actual=blocks[0].in_degrees().to(torch.int64)
            groups=blocks[0].dstdata[DIGIT_SAMPLED_GROUPS].sum()*group_size if bundle else target.new_zeros(())
            coverage.append(torch.stack([target.sum(),actual.sum(),(actual<target).sum(),(target-actual).sum(),groups,(actual>target).sum()]))
            shapes.append(dict(input_nodes=len(inp),output_nodes=len(out),block_edges=[b.num_edges() for b in blocks]))
            total_rows+=len(inp);window_rows+=len(inp)
            if smoke:
                record.update(epoch=epoch,batch=i,loss=float(loss),parameters_after=model_hash(model));audits.append(record)
            if (i+1)%100==0 or i+1==limit:
                torch.cuda.synchronize();stop=time.perf_counter();current=snapshot(loader)
                cov=torch.stack(coverage).sum(0).cpu().tolist();coverage=[];assert cov[-1]==0
                windows.append(dict(start_update=1 if not windows else windows[-1]['end_update']+1,end_update=i+1,
                    wall_seconds=stop-window_start,input_nodes=window_rows,target_edges=cov[0],actual_edges=cov[1],
                    underfilled_owners=cov[2],shortfall_edges=cov[3],group_edges=cov[4],**interval(previous,current,window_rows)))
                previous=current;window_rows=0;window_start=time.perf_counter()
                if not smoke:progress(stage='training',arm=arm,seed=value,epoch=epoch+1,updates=i+1)
        torch.cuda.synchronize();train_seconds=time.perf_counter()-epoch_start
        values=torch.stack(losses).cpu().numpy();assert np.isfinite(values).all() and all(torch.isfinite(p).all() for p in model.parameters())
        assert len(shapes)==limit and (smoke or sum(s['output_nodes'] for s in shapes)==len(train_ids))
        assert smoke or shapes[-1]['output_nodes']==len(train_ids)-(limit-1)*batch_size
        if bundle:
            storage={k:v.data_ptr() for k,v in sampler._cuda_metadata['cuda:0'].items()}
            if metadata is None:metadata=storage
            assert metadata==storage and sampler._cuda_call_counter==(epoch+1)*limit
        validate_region(interval(begin,previous,total_rows))
        ev=evaluation.evaluate('valid',model,loader,bundle,labels,2 if smoke else None,features if smoke else None)
        assert ev['examples']==(batch_size*2 if smoke else valid_count) and ev['batches']==(2 if smoke else math.ceil(valid_count/batch_size))
        report=dict(epoch=epoch+1,updates=limit,root_sha256=root_hash,train_seconds=train_seconds,order_seconds=order_seconds,
                    losses=values.tolist(),loss_sha256=digest(values),shapes=shapes,windows=windows,training=interval(begin,previous,total_rows),
                    validation=ev,model_sha256=model_hash(model),metadata_reused=True,cache_object_reused=True,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if selected_reference is not None:
            ref=selected_reference['epochs'][epoch]
            report['selected_source_bridge']={
                'initial':initial==selected_reference['initial_parameters_sha256'],
                'roots':root_hash==ref['root_sha256'],
                'losses':report['losses']==ref['losses'],
                'model':report['model_sha256']==ref['model_sha256'],
                'validation_predictions':ev['prediction_sha256']==ref['validation']['prediction_sha256']}
            require(all(report['selected_source_bridge'].values()),'Native GIDS differs from selected source-only epoch '+str(epoch+1))
        epochs.append(report);write(output.parent/('epoch_%02d.json'%(epoch+1)),report)
        progress(stage='epoch_complete',arm=arm,seed=value,epoch=epoch+1,valid_accuracy=ev['accuracy'],train_seconds=train_seconds,smoke=smoke)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output.parent/'final_model.pt')
    test_result=None
    if not smoke:
        before_test=model_hash(model)
        with (output.parent/'test_invocation.json').open('x') as guard:
            __import__('json').dump(dict(checkpoint_epoch=total_epochs,model_sha256=before_test,checkpoint_sha256=sha(output.parent/'final_model.pt'),seed=value,arm=arm),guard)
        test_result=evaluation.evaluate('test',model,loader,bundle,labels)
        require(model_hash(model)==before_test,'Final test changed model')
        assert test_result['examples']==test_count and test_result['batches']==math.ceil(test_count/batch_size)
    evaluation_summary=evaluation.finish(total_epochs)

    end_cache=snapshot(loader)
    for point in (start_cache,end_cache):
        if 'native' in point:point['native']=point['native'].tolist()
    return dict(metadata_setup_seconds=metadata_setup_seconds,optimizer_effective={k:v for k,v in optimizer.param_groups[0].items() if k!="params"},evaluation_lifecycle=evaluation_summary,passed=True,arm=arm,seed=value,smoke=smoke,epochs=epochs,updates=sum(item["updates"] for item in epochs),
        initial_parameters_sha256=initial,initial_dgl_rng=initial_rng,final_parameters_sha256=model_hash(model),
        all_epochs_metadata_and_cache_reused=True,audits=audits,test=test_result,
        initial_cache=start_cache,final_cache=end_cache,training_seconds=sum(r['train_seconds'] for r in epochs),
        validation_seconds=sum(r['validation']['seconds'] for r in epochs),source_only=isinstance(loader,SourceLoader),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
