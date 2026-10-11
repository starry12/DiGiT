"""End-to-end stream/compact/grouped-host CPU test against int64 reference."""
import hashlib
import numpy as np
from .common import cpu_gate,require,write_new,digest
from .protocol import fixture_plan
from .data import fixture_inputs
from .stream import write_fixture
from .compact import build,CSC
from .layout import Grouped
from .sampler import HostSampler,STORAGE_ROW
from .training import epoch


def exercise(output):
    cpu_gate()
    from candidates.uks_sage_epoch_v1.runtime import setup_sampling_imports
    setup_sampling_imports()
    import torch,dgl
    from digit.reorganization import reorganize_to_bundle
    from candidates.pa_sage_cache_policy_v1.selection import reverse_pagerank,topk
    from candidates.pa_sage_cache_policy_v1.oracle import FeatureOracle
    torch.set_num_threads(1);torch.set_num_interop_threads(1);dgl.utils.set_num_threads(1)
    require(not torch.cuda.is_initialized(),'CPU fixture started CUDA')
    p=fixture_plan();n=p['nodes'];features,labels,selected=fixture_inputs(p)
    # Incoming lists deliberately include duplicate self/nonself edges and
    # asymmetric links. They are not a tiny extract of the actual UKL graph.
    rows=[np.asarray([(v+d)%n for d in (0,0,1,2,3,5,7,11,17,1)],np.int64) for v in range(n)]
    binding=dict(schema='digit-ukl-incoming-stream-v1',nodes=n,edges=sum(map(len,rows)),orientation='incoming',
        source_sha256=hashlib.sha256(b'ukl-fixture-v1').hexdigest(),node_map_sha256=hashlib.sha256(b'identity-257').hexdigest())
    output.mkdir(parents=True,exist_ok=False)
    with (output/'input.stream').open('xb') as f:write_fixture(f,binding,rows)
    receipt=build(output/'input.stream',output/'csc',binding,fixture=True,checkpoint_owners=31)
    compact=CSC.open(output/'csc',fixture=True)
    # Independently derive the int64 expected CSC from logical incoming lists.
    reference_rows=[np.r_[r[r!=v],np.int64(v)] for v,r in enumerate(rows)]
    ptr=np.r_[0,np.cumsum([len(r) for r in reference_rows],dtype=np.int64)]
    idx=np.concatenate(reference_rows).astype(np.int64)
    reference=CSC(ptr,idx,fixture=True,reference=True)
    np.testing.assert_array_equal(compact.ptr,ptr);np.testing.assert_array_equal(compact.indices,idx)
    revpr=topk(reverse_pagerank(ptr,idx),p['cpu_cache_rows'])
    bundle=reorganize_to_bundle(ptr,idx,features,output/'layout',dataset_name='UKL',dataset_size='cpu_fixture',
        group_size=2,replication_ratio=.2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096,
        hot_nodes=revpr,seed=0,feature_batch_rows=31)
    require(bundle.num_groups>0,'Fixture has no groups')
    groups=Grouped(bundle);group_reference=Grouped(bundle,reference=True)
    profile_sampler=HostSampler(compact,p['fanouts'],groups=groups,seed=p['profile']['seed'])
    order=np.random.default_rng(p['profile']['seed']).permutation(selected);counts=np.zeros(n,np.int64)
    for i in range(p['profile']['batches']):
        inp,_,_=profile_sampler.sample_blocks(order[i*8:(i+1)*8]);counts[inp.numpy()]+=1
    freq=topk(counts,p['cpu_cache_rows']);reports={}
    for arm in ('gids','digit'):
        cpu_gate();g=groups if arm=='digit' else None;gref=group_reference if arm=='digit' else None
        if arm=='digit':
            storage=bundle.arrays['storage_to_node'];payload=bundle.arrays['reordered_features'];hot=freq
        else:
            storage=np.r_[np.arange(n),np.full((-n)%8,-1)].astype(np.int64)
            payload=np.zeros((len(storage),128),np.float32);payload[:n]=features;hot=revpr
        oracle=FeatureOracle(storage,payload,features,hot,p['gpu_cache_bytes'])
        def cached(inp,blocks):return torch.from_numpy(oracle.fetch(inp.numpy(),blocks[0].srcdata[STORAGE_ROW].numpy()))
        direct=lambda inp,blocks:torch.from_numpy(features[inp.numpy()].copy())
        actual=epoch(p,HostSampler(compact,p['fanouts'],groups=g),selected,labels,cached)
        expected=epoch(p,HostSampler(reference,p['fanouts'],groups=gref),selected,labels,direct)
        for key in ('initial_model_sha256','final_model_sha256','root_sha256','sample_trace_sha256','losses'):
            require(actual[key]==expected[key],'Compact/cache changed '+arm+'/'+key)
        metrics=oracle.report();require(metrics['cpu_served_rows']>0 and metrics['ssd_served_rows']>0,'Cache routes unexercised')
        reports[arm]=dict(actual,arm=arm,oracle=metrics,compact_equals_int64_direct=True)
        write_new(output/(arm+'.json'),reports[arm])
    require(reports['gids']['root_sha256']==reports['digit']['root_sha256'] and
            reports['gids']['initial_model_sha256']==reports['digit']['initial_model_sha256'],'Pair roots/model mismatch')
    result=dict(passed=True,nodes=n,edges=compact.edges,groups=bundle.num_groups,storage_rows=len(bundle.arrays['storage_to_node']),
        compact_array_bytes=compact.ptr.nbytes+compact.indices.nbytes,full_eid_array_bytes=0,
        legacy_int64_csc_eids_bytes=ptr.nbytes+16*len(idx),compact_group_bytes=groups.array_bytes,
        reference_group_bytes=group_reference.array_bytes,updates_per_arm=4,examples_per_arm=25,last_batch=1,
        per_arm_trace_loss_model_exact=True,profile=dict(seed=23,batches=2,optimizer_updates=0,counts_sha256=digest(counts)),
        native_execution=False,gpu_queries=False,cuda_initialized=torch.cuda.is_initialized(),raw_ssd_access=False,
        full_dgl_graph_constructed=False,note='Actual CPU SAGE correctness. No GPU/UVA, decoder-build or native performance acceptance.')
    require(not torch.cuda.is_initialized(),'CPU fixture initialized CUDA')
    write_new(output/'summary.json',result);return result
