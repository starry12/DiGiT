from candidates.unified_training_v2.accounting import validate_counts
def counters(r,groups,edges,arm):
    commands=int(r['primary_commands']);bytes_=int(r['primary_bytes'])
    pairs=(bytes_//4096)-commands
    singles=commands-pairs
    if bytes_%4096 or min(pairs,singles)<0:raise ValueError('Invalid IG 4K/8K mix')
    gpu=int(r['gpu_hits'])
    requests=int(r['gpu_requests'])
    misses=int(r['gpu_inserts'])+int(r['gpu_partial_fills'])
    if (requests!=int(r['feature_gpu_ssd']) or requests!=gpu+misses
            or int(r['useful']['gpu_feature_rows'])!=requests
            or int(r['rows'])!=int(r['feature_cpu'])+requests):
        raise ValueError('IG native CPU/GPU/SSD request counters do not reconcile')
    c=dict(logical_rows=int(r['rows']),cpu_rows=int(r['feature_cpu']),gpu_rows=gpu,
           ssd_rows=misses,
           primary_single_commands=singles,primary_pair_commands=pairs,primary_bytes=bytes_,
           replay_commands=int(r['replay_commands']),replay_bytes=int(r['replay_bytes']),
           completed_commands=int(r['device_commands']),completed_bytes=int(r['device_bytes']),
           sampled_group_edges=int(groups),sampled_outer_edges=int(edges))
    validate_counts(c,4096,arm);return c
