"""UKS first complete performance epoch; preparation and native readiness are separate."""
import copy
from .common import ROOT,HERE,LARGE,ARMS,read,sha,require


def compile_plan():
    old=ROOT/'candidates/uks_sage_v2/protocol.json';p=copy.deepcopy(read(old))
    p.update(schema='digit-uks-sage-epoch-adapter-v1',reference_protocol=str(old),reference_protocol_sha256=sha(old),
        epochs=1,evaluation='disabled',repeats=1,arms={
            'gids':dict(sampler='dgl_neighbor',layout='primary',cpu_selection='revpr',gpu_policy='legacy'),
            'digit':dict(sampler='group_aware_outer',layout='g2_r20',cpu_selection='freq',gpu_policy='fifo')})
    p['systems']=list(ARMS)
    p['measurement']=dict(kind='complete_first_epoch',warmup_batches=0,
        updates=(p['training']['train_nodes']+p['batch_size']-1)//p['batch_size'],
        training_examples=p['training']['train_nodes'],last_batch=p['training']['train_nodes']%p['batch_size'],
        window_updates=100,validation_batches=0,test_calls=0,accuracy_claim=False,steady_state_claim=False,
        separate_costs=['graph/feature preparation','profile/TopK','cache installation','root order','model setup'],
        native_short_updates=4,all_smokes_before_any_full=True)
    p['training']['split']='uniform 10% without replacement, sorted int64; full independent epoch permutation'
    p['features']=dict(mode='logical_synthetic',dimension=256,row_bytes=1024,page_bytes=4096,
        rows_per_page=4,replicas='same logical node means bit-identical feature values',raw_ssd_writes=False,
        shared_proxy=dict(implemented=False,approved_for_uks=False,requires='UKS 256D pilot plus compatible cache semantics',
            logical_aliasing_allowed=False))
    p['profile']=dict(seed=23,batches=100,feature_reads=0,optimizer_updates=0,evaluation_calls=0,
        source='independent grouped sampling on fixed g2/r20; unique outer logical inputs per batch')
    p['layout']=dict(group_size=2,replication_ratio=.2,page_bytes=4096,minimum_transfer_bytes=4096,
        cache_slot_bytes=4096,hot_exclusion='fixed RevPR 10% on original normalized graph',
        group_values='two 1024-byte member rows per 4096-byte slot; remaining rows are padding')
    p['cache_selection']=dict(gids='reverse PageRank on original normalized graph, damping .85, 20 iterations',
        digit='independent grouped-sampler frequency, seed23, 100 batches',ties='descending score, ascending logical node ID',
        cpu_budget='same exact 10% logical rows; replicas alias one CPU slot',gpu_budget='same 4 GiB: GIDS legacy and DiGiT FIFO')
    p['graph_contract']=dict(normalized_edges=None,source_eid_dtype='int64',indptr_dtype='int64',
        sampling_mode='gpu_i32_uva_eid64',node_and_storage_dtype='int32 after range check',
        reverse_edges_added=False,eids='stable CSC-position int64 IDs; distinct even for duplicate edges',
        maximum_normalized_edges=p['source_edges']+p['nodes'])
    p['paths']=dict(source=p['paths']['source_default'],data=str(LARGE),original_grid='results/pa_sage_layout_continue_20260925_v4')
    p['preparation']=dict(builder='partitioned C++ stable CSC + compiled g2',
        grouping_rng='mt19937_64 seed0, unbiased Fisher-Yates; not byte-identical PA NumPy grouping',
        output_root=str(LARGE),filesystem_only=True,raw_ssd_writes=False,
        phases=['csc','rank','g2','synthetic','payload'],source_bucket_nodes=1048576,
        available_host_min_bytes=128*2**30,free_disk_min_bytes=1536*2**30,
        limits_scope='filesystem preparation only; native admission still pending')
    p['host_required_bytes']=None;p['heavy_stage_free_disk_bytes']=None;p['ssd_offsets']=None
    p['reconstruction_choices']=[
        '19 classes, Adam lr=.001/weight_decay=.001/dropout=.2 inherited from UKS v2; not paper-specified UKS hyperparameters.',
        'Equal GPU and CPU capacity; GIDS path uses native legacy replacement/DGL/RevPR, DiGiT grouped/Freq/FIFO. Shared exact-row adapter is reconstructed, not unmodified upstream GIDS.',
        'Fixed RevPR exclusion for g2 construction before independent grouped profiling; no measured-trace cache fitting.',
        'Preserve source nonself multiedges, replace self loops by one per node, do not automatically add reverse edges.',
        'Logical synthetic feature reference only; shared physical-row proxy remains deferred and cannot alias unequal replicas.',
        'One first epoch, no warmup or evaluation; not a reproduction of the paper 20-epoch average.']
    p['execution']=dict(native_ready=False,automatic_start=False,backend_abi=4,monitor='nvidia-smi',
        query_timeout_seconds=5,max_gap_enforced=False,fresh_process_per_arm=True,
        heavy_work_requires_grid_complete=True,large_preparation_implemented=True,
        remaining=['scalable CSC/g2 builder binding and actual metadata/feature receipts','full-graph RevPR and independent profile',
            'UKS device/host budget from final layout','isolated CUDA build and native short acceptance','native controller deployment'])
    return p


def validate(p):
    require(p==compile_plan(),'UKS protocol differs from the frozen complete-epoch plan')
    return p


def fixture_plan(nodes=257,batch_size=8):
    require(type(nodes) is int and 40<=nodes<=4096 and 1<=batch_size<=128,'Bounded fixture only')
    p=compile_plan();p['fixture']=True;p['nodes']=nodes;p['source_edges']=None;p['batch_size']=batch_size
    p['cpu_cache_rows']=nodes//10;p['training']['train_nodes']=nodes//10
    t=p['training']['train_nodes'];p['measurement'].update(updates=(t+batch_size-1)//batch_size,training_examples=t,last_batch=t%batch_size)
    p['profile']['batches']=min(2,p['measurement']['updates']);p['gpu_cache_bytes']=2*4096
    return p
