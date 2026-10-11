"""Explicit persistent allocations, training transients and 4 GiB headroom."""
import copy
from .common import *
def estimate(arm='digit_full'):
    from candidates.pa_sage_revpr_node_v1.admission import estimate as original
    base=copy.deepcopy(original());p=cfg();c=base['components_bytes']
    rows=read(ROOT/p['base_layout']/'final/bundle/manifest.json')['feature']['num_storage_rows'] if arm=='digit_full' else ((p['graph']['nodes']+7)//8)*8
    metadata=('reordered_indptr','reordered_indices','group_members','group_storage_base','supernode_to_group','node_to_primary_row')
    if arm=='gids':
        for key in metadata:c[key]=0
    c['bam_storage_range_allowance']=rows*32
    c['exact_cpu_row_map']=rows*4
    c['cpu_preload_indices']=p['cpu_cache_rows']*8
    c['useful_io_bitmap_and_counters']=p['gpu_cache_bytes']//512*8+48
    c['native_setup_allowance']=512*2**20
    c['pair_scratch_and_hash_allowance']=128*2**20
    # No evaluation, prefetch, accumulation, fused graph or overlap buffers.
    phases=dict(metadata=sum(c[k] for k in metadata),
        native=sum(c[k] for k in ('cache_data','cache_metadata_allowance','bam_storage_range_allowance',
                                 'useful_io_bitmap_and_counters','native_setup_allowance','pair_scratch_and_hash_allowance')),
        cpu_map=c['exact_cpu_row_map']+c['cpu_preload_indices'],degree=c['common_degree_counter'],
        model=c['model_optimizer_grad']+c['labels_masks_seed_indices'])
    headroom=c['runtime_reserve']+c['batch_features_activations']+c['sampled_edges_and_sort_scratch']
    base.update(arm=arm,required_bytes=sum(phases.values())+headroom,phase_allocations=phases,
        phase_headroom_bytes=headroom,explicit_runtime_reserve_bytes=c['runtime_reserve'],page_bytes=512,
        storage_rows=rows,protocol_sha256=sha(P),safety_multiplier=1.,formal_bound=False,
        allocation_policy='Explicit persistent allocations plus training transients plus 4GiB reserve; live phase checks')
    return base
def stage_check(folder,plan,stage,remaining):
    from candidates.pa_sage_mixed_bfs_v5.admission import stage_check as check
    check(folder,plan,stage,remaining)
