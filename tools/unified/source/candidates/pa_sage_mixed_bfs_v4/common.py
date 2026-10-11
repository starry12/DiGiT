"""BFS/decay integration, independently bound over the accepted mixed backend."""
from candidates.pa_sage_mixed_group_v1.common import *
from candidates.pa_sage_mixed_group_v1 import common as reference
from candidates.pa_sage_512b_pair_v4 import common as gids_reference
from candidates.pa_sage_mixed_bfs_v3 import common as predecessor

HERE=ROOT/'candidates/pa_sage_mixed_bfs_v4'
OUT=ROOT/'results/pa_sage_mixed_bfs_20261010_v4'
MODULE='candidates.pa_sage_mixed_bfs_v4'
VARIANTS=('gids_ae','random_ae','bfs64_ae')
GIDS_BINARY=gids_reference.BINARY


def binary_for(variant):return GIDS_BINARY if variant=='gids_ae' else BINARY


def verify():
    m=read(HERE/'manifest.json')
    require(predecessor.verify()==m['predecessor_sha256'],'Mixed BFS predecessor changed')
    for rel,digest in m['files'].items():require(sha(HERE/rel)==digest,'New source changed: '+rel)
    for path,digest in m['dependencies'].items():require(sha(path)==digest,'Dependency changed: '+path)
    require(sha(OUT/'inputs.json')==m['inputs_sha256'],'Input binding changed')
    p=read(HERE/'protocol.json');old=read(predecessor.HERE/'protocol.json')
    changed={'scope','edges','graph','graph_limit','optimizer','optimizer_origin','lr_schedule',
             'variants','full_order','evaluation','digit_cpu_hotset','admission_policy','gpu_label_storage'}
    require({k:v for k,v in p.items() if k not in changed}=={k:v for k,v in old.items() if k not in changed},'Unrelated setting changed')
    require(p['edges']==3342431700 and p['optimizer']==dict(lr=.001,weight_decay=0.) and p['lr_schedule']==dict(milestones=[],gamma=1.),'AE alignment changed')
    require(sha(OUT/'inputs.json')==sha(predecessor.OUT/'inputs.json'),'Training inputs changed from v3')
    return sha(HERE/'manifest.json')


def check_inputs(binding):
    from candidates.pa_sage_mixed_group_v1.identity import verify_input
    reference.check_inputs(read(reference.OUT/'inputs.json'))
    require(binding==read(OUT/'inputs.json'),'Wrong input binding')
    registry=read(HERE/'identity_registry.json')['files']
    require(set(registry)==set(binding['overlay_files']),'Wrong overlay registry')
    for path,desc in binding['overlay_files'].items():
        verify_input(path,desc['identity'],registry[path],identity)
    require(sha(binding['prepared'])==binding['prepared_sha256'],'Overlay evidence changed')


def full_schedule():return list(VARIANTS)


def validate_epoch(r):
    # Project only the protocol/mode labels for unchanged single-epoch training
    # gates. Actual epoch/schedule/order and mixed-command gates remain explicit.
    require(r['protocol']==read(HERE/'protocol.json'),'Wrong AE protocol')
    gids=r['arm']=='gids'
    if gids:
        require(r['variant']=='gids_ae' and r['read_mode']=='single' and not r['mixed_io'],'Wrong GIDS read mode')
        require(r['model_implementation']=='standard_sage' and r['group_mode']=='standard_uniform' and r['io_mode']=='legacy','Baseline implementation changed')
    if gids and r['device_io']['completed_bytes']>0:
        gids_reference.validate_completion(dict(r,protocol=read(gids_reference.HERE/'protocol.json')))
    elif r['device_io']['completed_bytes']>0:
        reference.reference.validate_completion(dict(r,variant='legacy',mixed_io=False,
            protocol=read(reference.reference.HERE/'protocol.json')))
    else:
        # A retained GPU cache may serve a complete later epoch. Do not invent
        # an SSD read to satisfy the original cold-one-epoch validator.
        validate_warm_epoch(r)
    reference.check_pair_counts(r['pair_counts'],r['device_io'],not gids,require_pair=False)
    require(r['root_order_exact'],'Training ignored explicit root order')


def validate_warm_epoch(r):
    p=read(HERE/'protocol.json');n=p['smoke_examples'] if r['smoke'] else p['train_count']
    updates=(n+p['batch_size']-1)//p['batch_size'];pipe=r['pipeline'];io=r['device_io'];gpu=r['gpu_cache']
    require(r['epoch']>1 and r['passed'] and r['finite_parameters'],'Invalid warm epoch')
    require(r['examples']==n and r['updates']==updates and r['root_coverage_exact'],'Incomplete warm training')
    require(r['tail_size']==n-(updates-1)*p['batch_size'],'Lost tail')
    require(pipe['drained'] and pipe['sampled_batches']==pipe['read_batches']==pipe['delivered_batches']==updates,'Undrained warm pipeline')
    gids=r['arm']=='gids'
    require(r['sample_count']==r['fetch_count']==updates and r['native_grouped_batches']==(0 if gids else updates),'Wrong sampler/fetch coverage')
    require(sum(r['feature_access'].values())==r['logical_feature_requests'],'Feature count differs')
    require(gpu['requests']==r['feature_access']['gpu_ssd']==gpu['hits'] and gpu['inserts']==0,'Unaccounted warm-cache miss')
    require(io['outstanding']==io['submitted_commands']==io['completed_commands']==io['completed_bytes']==0,'Unexpected warm I/O')
    require(gpu['capacity_pages']==8388608 and gpu['policy']==('legacy' if gids else 'fifo') and r['cpu_cache_rows']==p['cpu_rows'],'Cache configuration differs')
    require(r['page_bytes']==512 and r['window_buffer'] and r['accumulator'] and r['block_index_bits']==32,'Read/block mode differs')
    if not gids:require(r['cpu_cache']['replica_aliases'] and r['cpu_cache']['padding_uncached'],'CPU map differs')
    if gids:
        require(pipe['limit_batches']==4 and pipe['limit_rows']==262144 and pipe['max_merged_batches']<=4 and pipe['max_pending_batches']<=5,'GIDS accumulator limits differ')
        require(math.isfinite(r['timing']['e2e_seconds']) and r['timing']['e2e_seconds']>0,'Invalid GIDS timing')
        for phase in ('at_training_start','at_training_end'):
            masks=r['affinity'][phase]
            require(masks and all(v==[p['bound_logical_cpu']] for v in masks.values()),'GIDS affinity differs')
        if r['smoke']:require(r['smoke_checks']['features_bit_exact'] and r['smoke_checks']['sampled_edges_exact'],'Missing GIDS audit')
        return
    require(pipe['max_outstanding_reads']<=1 and pipe['submitted_reads']==pipe['collected_reads']==pipe['read_calls'] and r['async_backend']['outstanding']==0,'Pending async read')
    dense=r['dense_graph'];full=sum(s['output_nodes']==1024 for s in r['shapes'])
    require(dense['graph_forwards']==full and dense['eager_forwards']==updates-full and dense['capture_count']==1,'Dense replay coverage differs')
    require(dense['prepared'] and dense['optimizer_identity_preserved'] and dense['parameter_identity_preserved'],'Dense state differs')
    require(r['io_mode']=='overlap' and r['group_mode']=='incremental' and r['mean_mode']=='legacy','Optimizations changed')
    require(r['sampler_binary_sha256']==sha(SAMPLER_BINARY) and r['normalization_binary_sha256']==sha(NORMALIZATION_BINARY),'Wrong math binaries')
    require(math.isfinite(r['timing']['e2e_seconds']) and r['timing']['e2e_seconds']>0,'Invalid warm timing')
    for phase in ('at_training_start','at_training_end'):
        masks=r['affinity'][phase]
        require(masks and all(v==[p['bound_logical_cpu']] for v in masks.values()),'Affinity differs')
    if r['smoke']:require(all(r['smoke_checks'].values()),'Missing warm feature/graph audit')


def validate_completion(r):
    from .order import learning_rate
    p=read(HERE/'protocol.json');n=p['smoke_epochs'] if r['smoke'] else p['epochs']
    require(r['passed'] and r['protocol']==p and r['variant'] in VARIANTS,'Wrong experiment')
    require(len(r['epochs'])==n and r['objects_reused'],'Incomplete continuous training')
    require(r['label_storage']==dict(dtype='uint8',bytes=p['nodes']),'Wrong compact label report')
    for i,e in enumerate(r['epochs']):
        validate_epoch(e)
        require(e['epoch']==i+1 and e['learning_rate']==learning_rate(p,i),'Incorrect LR schedule')
        require(e['validation']['passed'] and e['validation']['model_unchanged'] and e['validation']['rng_unchanged'],'Invalid validation')
        require(e['candidate_sha256']==r['candidate_sha256'] and e['binary_sha256']==r['binary_sha256'],'Epoch source differs')
        require(e['epoch_initial_model_sha256']==(r['initial_model_sha256'] if i==0 else r['epochs'][i-1]['final_model_sha256']),'Model reset between epochs')
    require(r['binary_sha256']==sha(binary_for(r['variant'])),'Wrong native backend')
    if r['variant']=='gids_ae':
        require(sum(e['pair_counts'][0] for e in r['epochs'])>0,'No real GIDS reads')
    else:require(sum(e['pair_counts'][1] for e in r['epochs'])>0,'No real grouped 1KiB reads')
    require(r['evaluation_calls']==dict(valid=n,test=0 if r['smoke'] else 1),'Wrong evaluation count')
    if not r['smoke']:
        require(r['test']['passed'] and r['test']['model_unchanged'] and r['test']['rng_unchanged'],'Invalid final test')
        require(r['final_model_sha256']==r['epochs'][-1]['final_model_sha256']==r['test']['model_sha256'],'Test model is not epoch20')
    return True
