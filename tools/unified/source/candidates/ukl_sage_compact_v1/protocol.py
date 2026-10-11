"""Explicit first-epoch reconstruction choices; no native admission yet."""
import copy
from .common import ROOT, LARGE, require


def compile_plan():
    n, e, train = 787801471, 47614527250, 78780147
    return dict(schema='digit-ukl-compact-protocol-v1', dataset='UKL', nodes=n, source_edges=e,
        feature_dim=128, feature_dtype='float32', classes=19, model='sage', layers=3, hidden=128,
        fanouts=[10,5,5], batch_size=1024, dropout=.2, seed=0,
        optimizer=dict(name='Adam', lr=.001, weight_decay=.001), epochs=1, evaluation='disabled', repeats=1,
        training=dict(train_nodes=train, split_seed=0, order_seed=0, split='uniform 10% without replacement; sorted int64'),
        measurement=dict(kind='complete_first_epoch', updates=76934, last_batch=755,
            warmup_batches=0, validation_batches=0, test_calls=0, accuracy_claim=False, steady_state_claim=False,
            separate_costs=['decode/CSC/layout','presampling/ranking','cache preload','root order','model initialization']),
        synthetic=dict(feature_seed=20260922, label_seed=20260923, block_rows=65536,
            generator='PCG64 SeedSequence([seed,block_index])', feature_distribution='uniform[-1,1) float32'),
        features=dict(mode='logical_synthetic_reference', row_bytes=512, page_bytes=4096, rows_per_page=8,
            replicas='bit-identical per logical node', shared_proxy_accepted=False, raw_ssd_writes=False),
        graph=dict(direction='incoming', normalization='preserve nonself multiedges; one self loop per node; no added reverse edges',
            neighbor_dtype='int32', offset_dtype='int64', eid_dtype='int64', eid_storage='implicit CSC position',
            storage_row_dtype='int64', reordered_graph_location='host reference; native UVA/partition kernel deferred',
            full_dgl_graph=False, full_gpu_metadata=False),
        layout=dict(group_size=2, replication_ratio=.2, page_bytes=4096,
            hot_exclusion='fixed RevPR 10% before independent grouped profile'),
        arms=dict(gids=dict(sampler='uniform incoming occurrence', cpu_selection='RevPR'),
                  digit=dict(sampler='outer grouped PPS without replacement; member costs 1/2', cpu_selection='Freq')),
        cpu_cache_rows=train, gpu_cache_bytes=4*2**30, gpu_policy='fifo',
        profile=dict(seed=23, batches=100, optimizer_updates=0, feature_reads=0),
        stream=dict(schema='digit-ukl-incoming-stream-v1', orientation='incoming',
            producer='bound transpose BVGraph or externally transposed original; never relabel outgoing as incoming',
            source_and_node_map_hashes_required=True, frame='owner i32 BE; degree i64 BE; neighbors i32 BE',
            chunk_items=65536, footer='DIGEND01 plus edge_count i64 BE', checkpoint='owner boundary; source replay on resume'),
        paths=dict(source='/mnt/n3/uk2014', large_output=str(LARGE)),
        execution=dict(native_ready=False, automatic_start=False, large_preparation_accepted=False,
            heavy_work_requires_grid_complete=True, monitor='nvidia-smi', query_timeout_seconds=5,
            max_gap_enforced=False, host_required_bytes=None, gpu_required_bytes=None,
            gpu_sampler_implemented=False, decoder_compiled=False),
        reconstruction_choices=[
            '19 classes and optimizer/dropout choices inherited from UKS; not paper-specified UKL hyperparameters.',
            'Both reconstructed arms use same logical CPU capacity and 4 GiB FIFO; not unmodified upstream GIDS.',
            'CPU reference uses seeded PCG64 choices; distribution preserved, not legacy Torch/CUDA bitstream parity.',
            'Logical synthetic reference tests do not accept a physical-row proxy with unequal replicas.',
            'One complete first epoch is not the paper 20-epoch average.'])


def validate(p):
    require(p == compile_plan(), 'Protocol differs from the frozen UKL plan')
    return p


def fixture_plan():
    p = copy.deepcopy(compile_plan());p.update(fixture=True, nodes=257, source_edges=None, batch_size=8,
                                             cpu_cache_rows=25, gpu_cache_bytes=8192)
    p['training']['train_nodes']=25;p['measurement'].update(updates=4,last_batch=1)
    p['profile']['batches']=2
    return p


def budget():
    p=compile_plan();n,e=p['nodes'],p['source_edges'];en=e+n
    return dict(formal_admission=False, compact_csc_upper_bytes=8*(n+1)+4*en,
        current_csc_eids_upper_bytes=8*(n+1)+16*en, cpu_feature_bytes=p['cpu_cache_rows']*512,
        original_feature_bytes=n*512, host_required_bytes=None, gpu_required_bytes=None,
        remaining='actual group/padding counts, host reordered graph, maps, native workspaces, preload and runtime peaks')
