from main_common import *

def run(args):
    import torch
    import runner as r
    from digit.gpu_admission import check_live
    output=Path(args.output)
    require(not output.exists(),'Output must be fresh')
    require(args.seed in (0,1,2),'The Papers protocol specifies seeds 0,1,2')
    execution=r.verify_execution(args.arm); check_device()
    output.parent.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter(); r.startup()
    plan=read(D/('admission.json' if args.model=='sage' else args.model+'_admission.json'))
    admission=check_live(plan);require(admission['passed'],'Insufficient GPU/host memory')
    if 'allocator_cap_bytes' in plan:
        torch.cuda.set_per_process_memory_fraction(plan['allocator_cap_bytes']/torch.cuda.get_device_properties(0).total_memory)
    def progress(**kw):
        write(output.parent/'progress.json',dict(updated=time.time(),**kw));print(json.dumps(kw),flush=True)
    r.progress=progress
    objects={}
    if args.model in ('gcn','gat'):
        from model_adapter import make_model,make_optimizer
        def setup(seed):
            r.seed(seed);model=make_model(args.model,'cuda',checked=False);optimizer=make_optimizer(model)
            objects.update(model=model,optimizer=optimizer)
            initial=r.model_hash(model);r.seed(seed);model.train();return model,optimizer,initial
        r.setup_model=setup
    if args.model=='gat':
        loop=ROOT/'ae/papers/gat_training_loop.py'
        exec(compile(loop.read_text(),str(loop),'exec'),r.__dict__)
    loader,bundle,cpu,inherited=r.initialize(args.arm)
    graph=r.PRIOR.graph_without_labels()
    labels=r.arr(read(P/'protocol.json')['dataset']['label_identity']['path']).reshape(-1)
    degrees=torch.from_numpy(r.arr(S/'original_degrees_i32.npy').copy()).cuda()
    setup_seconds=time.perf_counter()-began
    result=r.run_training(args.arm,args.seed,loader,bundle,graph,labels,degrees,output,smoke=False,features=None)
    if args.model=='gat':
        torch.save(objects['optimizer'].state_dict(),output.parent/'final_optimizer.pt')
        require(torch.cuda.mem_get_info()[0]>=plan['minimum_runtime_free_bytes'],'Runtime GPU memory floor violated')
    require(result['updates']==23580 and result['test'] is not None and not result['audits'],'Wrong formal scope')
    result.update(model=args.model,execution_sha256=execution,protocol_sha256=sha(P/'protocol.json'),setup_seconds=setup_seconds,
                  admission=admission,worker_seconds=time.perf_counter()-began,raw_writes=False)
    require(verify_release()==execution,'Release changed during run');write(output,result)
