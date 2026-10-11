from candidates.unified_training_v2.accounting import validate_counts

def counters(raw,groups,edges,arm):
    if not raw['reconciled']:raise ValueError('Unreconciled native interval')
    s=raw['serving'];d=raw['device'];z=raw['request_sizes']
    c=dict(logical_rows=s['logical_requests'],cpu_rows=s['cpu_served_rows'],gpu_rows=s['gpu_served_rows'],ssd_rows=s['ssd_served_rows'],
        primary_single_commands=z['primary_512b'],primary_pair_commands=z['primary_1k'],primary_bytes=d['primary_bytes'],
        replay_commands=d['replay_commands'],replay_bytes=d['replay_bytes'],completed_commands=d['completed_commands'],completed_bytes=d['completed_bytes'],
        sampled_group_edges=groups,sampled_outer_edges=edges)
    validate_counts(c,512,arm);return c
