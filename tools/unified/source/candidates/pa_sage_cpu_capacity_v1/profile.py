"""Independent seed-23 native sampling; no feature loader, optimizer or SSD reads."""
import hashlib
import time
import numpy as np
from .common import require, read, sha, write, identity, heavy_gate, verify, progress, complete_handshake
from .selection import select_all


def collect_batches(batches, counts, expected_batches):
    """Count unique outer input nodes once per batch, using caller-owned storage."""
    require(counts.ndim == 1 and counts.dtype == np.int64 and np.all(counts == 0), 'Frequency output must start empty')
    seen=total=0
    for ids in batches:
        require(seen < expected_batches, 'Presampling exceeded its frozen extent')
        ids=np.asarray(ids)
        require(ids.ndim == 1 and ids.dtype == np.int64 and len(ids)>0 and ids.min()>=0 and ids.max()<len(counts),
                'Bad presampling logical IDs')
        require(len(np.unique(ids)) == len(ids), 'Count unique feature inputs, not repeated edge occurrences')
        counts[ids]+=1;seen+=1;total+=len(ids)
    require(seen == expected_batches, 'Presampling ended early')
    return dict(batches=seen,total_logical_requests=total,optimizer_updates=0,evaluation_calls=0)


def execute(p, protocol_path, binding_path, output, large):
    heavy_gate();execution=verify()
    from .binding import check
    binding=read(binding_path);check(binding,protocol_path,execution)
    from .admission import estimate,live
    budget=live(estimate(p,binding['manifest'],profile=True));require(budget['passed'],'Profile resource admission failed')
    write(output/'admission.json',budget)
    from .native_support import setup_sampling_imports,startup,seed,graph_sampler,train_ids
    setup_sampling_imports()
    import torch
    from candidates.pa_sage_cache_policy_v1.training import trace_update
    from candidates.pa_sage_cache_policy_v1.common import digest
    from ae.pa_sage.observations import CheckpointObservations
    startup();observer=CheckpointObservations(output/'resources.json');observer.mark('profile_setup')
    seed(p['profile']['seed']);started=time.time()
    graph,arrays,bundle,sampler=graph_sampler(p,binding,p['profile']['seed'])
    roots=np.random.default_rng(np.random.SeedSequence([p['profile']['seed'],0])).permutation(train_ids(binding))
    count=p['profile']['batches'];size=p['batch_size']
    require(count*size<=len(roots),'Independent profile exceeds training split')
    selected=roots[:count*size].copy();root_gpu=torch.from_numpy(selected).cuda()
    large.mkdir(parents=True,exist_ok=False)
    path=large/'frequency_counts.npy'
    counts=np.lib.format.open_memmap(path,mode='w+',dtype=np.int64,shape=(p['graph']['nodes'],));counts[:]=0
    trace=hashlib.sha256();observer.mark('presampling')
    def batches():
        for i in range(count):
            target=root_gpu[i*size:(i+1)*size]
            inp,out,blocks=sampler.sample_blocks(graph,target)
            require(torch.equal(out,target),'Profile changed root order')
            trace_update(trace,inp,out,blocks)
            yield inp.detach().cpu().numpy().astype(np.int64,copy=False)
            progress(output,'presampling',batches_done=i+1,total_batches=count)
    presampling_started=time.perf_counter()
    extent=collect_batches(batches(),counts,count);counts.flush()
    presampling_seconds=time.perf_counter()-presampling_started
    require(sampler._cuda_call_counter==count,'Profile did not use the native sampler for every batch')
    selection_started=time.perf_counter()
    hot=select_all(counts,p,large)
    selection_seconds=time.perf_counter()-selection_started
    check(binding,protocol_path,execution);observer.mark('profile_complete')
    import sys
    require(not any(k.startswith('BAM_Feature_Store') for k in sys.modules),'Profile unexpectedly imported a feature backend')
    report=dict(kind='independent_native_presampling',passed=True,native=True,fixture=False,
        source_sha256=execution,protocol_sha256=sha(protocol_path),binding_sha256=sha(binding_path),
        graph_sha256=binding['graph_sha256'],layout_sha256=binding['layout_sha256'],
        seed=p['profile']['seed'],measurement_seed=p['seed'],fanouts=p['fanouts'],batch_size=size,
        root_sha256=digest(selected),sample_trace_sha256=trace.hexdigest(),
        counts=dict(path=str(path),sha256=sha(path),identity=identity(path)),hot=hot,
        raw_ssd_access=False,feature_reads=0,seconds=time.time()-started,admission=budget,
        presampling_and_count_flush_seconds=presampling_seconds,selection_and_seal_seconds=selection_seconds,
        resource_observations=observer.result(),**extent)
    complete_handshake(output,report)
