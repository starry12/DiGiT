"""Disjoint training/evaluation reserves, with 4 GiB explicit runtime headroom.

The previous budget added evaluation to the live training envelope and then
multiplied the entire allocation plus its reserves by 1.15. Here every original
component remains; training batches are released before evaluation, so use the
maximum of those phases. No extra multiplier is applied over the explicit
4 GiB runtime reserve. This is an engineering admission, not an OOM proof.
"""
from .common import *


def estimate(p,manifest,arm):
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import admission
    base=admission(p,manifest,arm);c=dict(base['components_bytes'])
    c.update(pair_scratch_and_hash_allowance=128*2**20 if arm=='digit' else 0,
             overlap_additional_batch_allowance=512*2**20 if arm=='digit' else 0,
             dense_graph_pool_allowance=256*2**20 if arm=='digit' else 0)
    metadata_keys=('reordered_indptr','reordered_indices','group_members','group_storage_base','supernode_to_group','node_to_primary_row')
    allocations=dict(metadata=sum(c[k] for k in metadata_keys),
        native=c['cache_data']+c['cache_metadata_allowance']+c['bam_storage_range_allowance']+c['pair_scratch_and_hash_allowance'],
        cpu_map=c['exact_cpu_row_map']+c['cpu_preload_indices'],labels=c['labels_masks_seed_indices'],
        dense=c['model_optimizer_grad']+c['dense_graph_pool_allowance'])
    train=c['batch_features_activations']+c['sampled_edges_and_sort_scratch']+c['window_retained_batches_allowance']+c['overlap_additional_batch_allowance']
    # Evaluation can use the same feature/activation envelope, without WB/overlap;
    # add 512 MiB for CPU-to-GPU evaluation staging instead.
    evaluation=c['batch_features_activations']+c['sampled_edges_and_sort_scratch']+512*2**20
    headroom=max(train,evaluation)+c['runtime_reserve']
    required=sum(allocations.values())+headroom
    return dict(base,components_bytes=c,phase_allocations=allocations,
        training_transient_bytes=train,evaluation_transient_bytes=evaluation,
        phase_headroom_bytes=headroom,required_bytes=required,safety_multiplier=1.,
        explicit_runtime_reserve_bytes=c['runtime_reserve'],formal_bound=False,
        allocation_policy='Persistent allocations plus max(training,evaluation) plus 4GiB; live checks at each stage')


def stage_check(folder,plan,stage,remaining):
    import torch
    torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
    reusable=max(0,reserved-allocated)
    need=plan['phase_headroom_bytes']+sum(plan['phase_allocations'][k] for k in remaining)
    record=dict(stage=stage,remaining=list(remaining),required_bytes=need,
        free_bytes=free,torch_allocated_bytes=allocated,torch_reserved_bytes=reserved,
        reusable_torch_bytes=reusable,total_bytes=total,passed=free+reusable>=need)
    write(folder/('memory_'+stage+'.json'),record)
    require(record['passed'],'GPU stage admission failed: '+stage)
