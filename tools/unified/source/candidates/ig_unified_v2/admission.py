import math
def estimate():
    from candidates.ig_demand_pair_v1.admission import estimate as parent
    r=parent();extra=405504*4096
    r['components_bytes']['one_extra_prefetched_feature_buffer']=extra
    r['components_bytes']['cuda_graph_reserve']=256*2**20
    r['required_bytes']=math.ceil(sum(r['components_bytes'].values())*1.2)
    r['host_required_bytes']+=8*2**30
    from .common import cfg
    from candidates.unified_training_v2.order import order_memory_bytes
    r['root_order_host_bytes']=order_memory_bytes(cfg()['train_nodes'],'gids')
    r['host_required_bytes']+=r['root_order_host_bytes']
    r['formal_bound']=False;r['reservation']=False
    return r
