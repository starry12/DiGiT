
from copy import deepcopy
from candidates.unified_training_v2.series import aggregate
from candidates.unified_training_v2.accounting import FIELDS,add_counts,metrics,validate_counts

def raw_counts(raw,groups,edges,arm,row_bytes=512):
    if not raw['reconciled']:raise ValueError('Unreconciled native interval')
    s=raw['serving'];d=raw['device'];z=raw['request_sizes']
    if row_bytes!=512:raise ValueError('Raw legacy converter requires explicit 512-byte geometry')
    gpu=s.get('gpu_served_rows',s.get('gpu_hit_rows'))
    if gpu is None or ('gpu_served_rows' in s and 'gpu_hit_rows' in s and s['gpu_served_rows']!=s['gpu_hit_rows']):
        raise ValueError('Missing or inconsistent GPU serving count')
    value=dict(logical_rows=s['logical_requests'],cpu_rows=s['cpu_served_rows'],gpu_rows=gpu,ssd_rows=s['ssd_served_rows'],
        primary_single_commands=z['primary_512b'],primary_pair_commands=z['primary_1k'],primary_bytes=d['primary_bytes'],
        replay_commands=d['replay_commands'],replay_bytes=d['replay_bytes'],completed_commands=d['completed_commands'],completed_bytes=d['completed_bytes'],
        sampled_group_edges=groups,sampled_outer_edges=edges)
    validate_counts(value,row_bytes,arm);return value

def summarize(config,receipts):
    from candidates.unified_training_v2.config import validate
    validate(config)
    base=aggregate(config,receipts)
    if config['data']['labels']=='synthetic' and base['accuracy_evaluated']:
        raise ValueError('Synthetic labels cannot establish accuracy')
    result=dict(base,schema='digit-unified-results-v2',row_bytes=config['data']['row_bytes'],
                root_order_comparison='GIDS_random_DiGiT_BFS256',feature_kind=config['data']['features'],label_kind=config['data']['labels'],
                profile_order=deepcopy(config['profile_order']),cache_profile=deepcopy(config['cache_profile']),
                short_profile_functional_only=config['cache_profile']['scope']=='short_window',
                arms={},epochs=[],source_sha256=receipts[0]['report']['initial']['source_sha256'])
    for arm in ('gids','digit'):
        reports=sorted((x['report'] for x in receipts if x['report']['arm']==arm),key=lambda r:r['repetition'])
        counts={k:0 for k in FIELDS};warm_included={k:0 for k in FIELDS};times=[]
        for r in reports:
            times.append(r['measured_seconds'])
            for e in r['epochs']:
                counts=add_counts(counts,e['counters']);warm_included=add_counts(warm_included,e['complete_counters'])
                result['epochs'].append(dict(arm=arm,round=r['repetition'],epoch=e['epoch'],seconds=e['seconds'],
                    training_wall_seconds=e['training_wall_seconds'],warmup_seconds=e['warmup_seconds'],
                    measured_batches=e['measured_batches'],measured_roots=e['measured_roots'],
                    counters=deepcopy(e['counters']),metrics=metrics(e['counters']),auxiliary=deepcopy(e['auxiliary'])))
        result['arms'][arm]=dict(per_round_seconds=times,training_seconds=sum(times),
            mean_training_epoch_seconds=sum(times)/sum(len(r['epochs']) for r in reports),
            measured_counters=counts,including_warmup_counters=warm_included,weighted_metrics=metrics(counts),
            cpu_feature_bytes=reports[0]['initial']['cpu_feature_bytes'],gpu_feature_bytes=reports[0]['initial']['gpu_feature_bytes'])
    a=result['arms']['gids']['measured_counters'];b=result['arms']['digit']['measured_counters']
    result['ssd_primary_bytes_reduction_fraction']=1-b['primary_bytes']/a['primary_bytes'] if a['primary_bytes'] else None
    result['notes']=['All requested rounds retained; five-round speedup is median of paired ratios, not ratio of medians.',
        'Service fractions use sums of requests, not averages of percentages. Warmup counters are separate.',
        'Sampled group edges, primary pair-command fraction and physical bytes are distinct metrics.',
        'Training times exclude preparation and evaluation; overlapped stage times are not added.',
        'One full20 pair uses ratio of 20-epoch time totals, not median epoch ratio.']
    if result['short_profile_functional_only']:
        result['notes'].append('320-batch profiling; RevPR-set fallback preserves capacity. Functional check, not full-epoch cache effectiveness evidence.')
    return result
