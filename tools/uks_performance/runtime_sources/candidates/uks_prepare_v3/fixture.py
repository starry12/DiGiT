"""Tiny real DGL/SAGE epochs, each compared with direct logical feature reads."""
import numpy as np
from .common import ARMS,require,write_new,cpu_gate
from .protocol import fixture_plan
from .data import fixture_inputs
from .graph import normalized_fixture,graph_from_csc
from .runtime import setup_sampling_imports
from .oracle import FeatureOracle


def exercise(output):
    cpu_gate();setup_sampling_imports()
    import torch,dgl
    from digit.reorganization import reorganize_to_bundle,assert_csc_adjacency_equivalent,expand_reorganized_csc
    from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW
    from candidates.pa_sage_cache_policy_v1.selection import reverse_pagerank,topk
    from candidates.pa_sage_cache_policy_v1.training import profile_cpu
    from .training import cpu_epoch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);dgl.utils.set_num_threads(1)
    require(not torch.cuda.is_initialized(),'CPU fixture started CUDA')
    p=fixture_plan();n=p['nodes'];features,labels,selected=fixture_inputs(p)
    raw=np.array([(i,(i+d)%n) for i in range(n) for d in (0,1,2,3,5,7,11,17)],dtype=np.int64).T
    raw=np.c_[raw,raw[:,::47]] # duplicate nonself edges + old self loops
    ptr,idx,eids=normalized_fixture(raw,n);graph=graph_from_csc(ptr,idx,eids,n,fixture=True)
    revpr=topk(reverse_pagerank(ptr,idx),p['cpu_cache_rows'])
    output.mkdir(parents=True,exist_ok=False)
    bundle=reorganize_to_bundle(ptr,idx,features,output/'layout',dataset_name='UKS',dataset_size='cpu_fixture',
        group_size=2,replication_ratio=.2,page_size=4096,minimum_transfer_bytes=4096,target_request_bytes=4096,
        hot_nodes=revpr,seed=0,feature_batch_rows=31)
    require(bundle.manifest['feature']['dim']==256 and bundle.io_geometry.feature_row_bytes==1024,'Layout did not retain UKS width')
    require(bundle.num_groups>0,'Fixture created no group nodes')
    def grouped(seed):return DiGiTNeighborSampler(p['fanouts'],bundle,cuda_mode='disabled',metadata_mode='cpu_eid',random_seed=seed)
    counts,profile=profile_cpu(grouped(23),graph,selected,n,23,p['profile']['batches'],p['batch_size'])
    freq=topk(counts,p['cpu_cache_rows'])
    storage=np.asarray(bundle.arrays['storage_to_node']);payload=np.asarray(bundle.arrays['reordered_features'])
    reports={};changed_routes=False
    for arm in ARMS:
        cpu_gate()
        if arm=='gids':
            mapping=np.r_[np.arange(n),np.full((-n)%4,-1)].astype(np.int64);primary=np.arange(n,dtype=np.int64)
            physical=np.zeros((len(mapping),256),np.float32);physical[:n]=features;hot=revpr
            new_sampler=lambda:dgl.dataloading.NeighborSampler(p['fanouts'])
        else:
            mapping=storage;primary=bundle.arrays['node_to_primary_row'];physical=payload;hot=freq
            new_sampler=lambda:grouped(0)
        oracle=FeatureOracle(mapping,physical,features,hot,p['gpu_cache_bytes'])
        def fetch(inp,blocks):
            logical=inp.numpy().astype(np.int64,copy=False)
            rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].numpy() if arm=='digit' else primary[logical]
            return torch.from_numpy(oracle.fetch(logical,np.asarray(rows,dtype=np.int64)))
        cached=cpu_epoch(p,new_sampler(),graph,selected,labels,fetch)
        direct=cpu_epoch(p,new_sampler(),graph,selected,labels,lambda inp,blocks:torch.from_numpy(features[inp.numpy()].copy()))
        for key in ('initial_model_sha256','final_model_sha256','root_sha256','sample_trace_sha256','losses','shapes'):
            require(cached[key]==direct[key],'256D cache changed training: '+arm+'/'+key)
        report=dict(cached,arm=arm,oracle=oracle.report(),matches_direct_logical_features=True)
        require(report['oracle']['cpu_served_rows']>0 and report['oracle']['ssd_served_rows']>0,'Fixture did not exercise CPU and SSD routes')
        write_new(output/(arm+'.json'),report);reports[arm]=report
    require(reports['gids']['root_sha256']==reports['digit']['root_sha256'] and
            reports['gids']['initial_model_sha256']==reports['digit']['initial_model_sha256'],'Pair changed roots/model initialization')
    # Grouped sampling can change samples relative to GIDS; only each arm's
    # cached/direct controls must match, not the two different samplers.
    result=dict(passed=True,fixture=True,native_execution=False,raw_ssd_access=False,cuda_initialized=False,
        nodes=n,graph_edges=len(idx),group_count=bundle.num_groups,feature_dim=256,storage_rows=len(storage),
        updates_per_arm=reports['gids']['updates'],examples_per_arm=len(selected),last_batch=reports['gids']['last_batch'],
        independent_profile=profile,per_arm_cache_vs_direct_exact=True,evaluation_calls=0,
        original_graph_normalization='source nonself multiedges retained; exactly one self loop; no automatic reverse edges',
        comparison_scope='Per-arm true CPU SAGE feature/cache correctness; no native performance or cross-sampler accuracy claim')
    require(not torch.cuda.is_initialized(),'Fixture opened CUDA')
    write_new(output/'summary.json',result);return result
