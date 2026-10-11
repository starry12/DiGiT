"""Explicit adapter: no installed AE or active worker code is modified."""
def sampler_class(use_cuda=False):
    from candidates.uks_mixed_1k2k_v1.runtime import setup_sampling_imports
    setup_sampling_imports()
    import digit.sampler as module
    if use_cuda:
        from .common import extension
        module._cuda_extension=extension()
    from .policy import select
    class GroupFirstSampler(module.DiGiTNeighborSampler):
        selection_policy='groups_first_without_replacement'
        def select_outer_units(self,candidates,fanout,device,choices=None):
            return select(candidates,fanout,device,num_nodes=self.num_nodes,choices=choices)
    return GroupFirstSampler

def graph_sampler(arm,seed,variant):
    # Reuse anonymous CSC copies and validated v1/v2 overlay loading; replace
    # only the sampler constructor during this explicit, single-worker call.
    from candidates.uks_group_compare_v2 import sampling as base
    from candidates.uks_mixed_1k2k_v1.runtime import setup_sampling_imports
    setup_sampling_imports()
    if arm=='gids':return base.graph_sampler(arm,seed,variant)
    import digit.sampler as module
    from candidates.uks_group_incremental_v1 import common as incremental
    from .common import extension
    old_class,old_extension=module.DiGiTNeighborSampler,incremental.extension
    cls=sampler_class(use_cuda=True)
    module.DiGiTNeighborSampler=cls;incremental.extension=extension
    try:return base.graph_sampler(arm,seed,variant)
    finally:module.DiGiTNeighborSampler=old_class;incremental.extension=old_extension
