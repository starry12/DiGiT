"""External PA full20 exporter; never change the active training candidate."""
import json
from pathlib import Path
from .report import raw_counts
from candidates.unified_training_v2.accounting import FIELDS,add_counts,metrics

def load(folder):
    folder=Path(folder)
    def read(name):return json.loads((folder/name).read_text())
    state=read('status.json')
    if not state.get('passed') or not state.get('complete') or state.get('stage')!='complete':
        return dict(schema='digit-unified-results-v2',dataset='PA',mode='full20',passed=False,complete=False,
            stage=state.get('stage'),error=state.get('error'),source=str(folder),speedup=None,
            reason='Wait for both complete workers and all validation/test acceptance; no partial speedup',
            completed=state.get('completed',[]))
    stages=['gpu_checks','smoke_gids','smoke_digit','gids','evaluate_gids','digit','evaluate_digit']
    if state['completed']!=stages:raise ValueError('Incomplete PA accepted stages')
    for stage in stages:
        if read(stage+'_exit.json')['returncode']!=0:raise ValueError('PA process exit failed')
    reports={a:read(a+'/report.json') for a in ('gids','digit')}
    accuracy={a:read('evaluate_'+a+'/report.json') for a in reports}
    schemas={r['protocol']['schema'] for r in reports.values()}
    if schemas=={'pa-bfs256-full20-v1'}:
        from candidates.pa_sage_bfs256_full20_v1.controller import aggregate
        order='GIDS_random_DiGiT_BFS256_same_full_training_set'
    elif schemas=={'pa-random-bfs-full20-v1'}:
        from candidates.pa_sage_random_bfs_full20_v1.controller import aggregate
        order='GIDS_random_DiGiT_BFS_same_full_training_set'
    else:raise ValueError('Unknown or mixed PA protocols')
    aggregate(reports,accuracy)
    from ae.common import sha
    for arm in reports:
        if accuracy[arm]['training_report_sha256']!=sha(folder/arm/'report.json'):raise ValueError('Accuracy/training provenance mismatch')
    result=dict(schema='digit-unified-results-v2',dataset='PA',mode='full20',passed=True,complete=True,
        source=str(folder),source_sha256=state['source_sha256'],evidence_scope='native',row_bytes=512,
        root_order_comparison=order,feature_kind='real',label_kind='real',
        accuracy_evaluated=True,full_epoch_training=True,arms={},epochs=[],stable_speedup_established=False,
        aggregation='single_pair_ratio_of_total_training_time')
    for arm,r in reports.items():
        count={k:0 for k in FIELDS}
        for e in r['epochs']:
            groups=sum(w['group_sampled_edges'] for w in e['windows']);edges=sum(w['outer_sampled_edges'] for w in e['windows'])
            c=raw_counts(e['training'],groups,edges,arm);count=add_counts(count,c)
            result['epochs'].append(dict(arm=arm,round=1,epoch=e['epoch'],seconds=e['seconds'],warmup_seconds=0,
                training_wall_seconds=None,measured_batches=e['updates'],measured_roots=e['examples'],counters=c,metrics=metrics(c),
                validation_accuracy=accuracy[arm]['validation'][e['epoch']-1]['accuracy']))
        result['arms'][arm]=dict(per_round_seconds=[r['measured_seconds']],training_seconds=r['measured_seconds'],
            mean_training_epoch_seconds=r['measured_seconds']/20,measured_counters=count,including_warmup_counters=count.copy(),
            weighted_metrics=metrics(count),cpu_feature_bytes=r['cache']['cpu_feature_bytes'],gpu_feature_bytes=r['cache']['gpu_feature_cache_bytes'],
            validation_accuracy=[v['accuracy'] for v in accuracy[arm]['validation']],test=accuracy[arm]['test'])
    a,b=result['arms']['gids'],result['arms']['digit'];result['speedup']=a['training_seconds']/b['training_seconds']
    result['paired_rounds']=[dict(round=1,gids_seconds=a['training_seconds'],digit_seconds=b['training_seconds'],speedup=result['speedup'])]
    x,y=a['measured_counters']['primary_bytes'],b['measured_counters']['primary_bytes'];result['ssd_primary_bytes_reduction_fraction']=1-y/x if x else None
    return result
