from .common import require,POLICIES

def graph_sampler(arm,seed,variant):
    require(variant in POLICIES,'Unknown sampler variant')
    require((arm=='gids')==(variant=='gids'),'Arm/variant mismatch')
    if variant=='first':
        from candidates.uks_group_first_v1.sampling import graph_sampler as sample
    else:
        from candidates.uks_group_compare_v2.sampling import graph_sampler as sample
    # Identical v2 adjacency for both DiGiT arms: only sampling policy changes.
    result=sample(arm,seed,'v2' if arm=='digit' else 'gids')
    if variant=='first':require(getattr(result[3],'selection_policy',None)==POLICIES['first'],'Priority sampler not installed')
    elif variant=='weighted':require(not hasattr(result[3],'selection_policy'),'Unexpected priority sampler in control')
    return result
