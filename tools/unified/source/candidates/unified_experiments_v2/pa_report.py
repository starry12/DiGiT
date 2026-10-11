"""Normalize complete PA short pairs without claiming matched examples."""
import statistics
from candidates.unified_training_v2.accounting import FIELDS,add_counts,metrics
from candidates.unified_results_v2.report import raw_counts
from .common import read
def summarize(mode,folders):
    from candidates.pa_sage_bfs256_short_v1.controller import aggregate
    from candidates.pa_sage_bfs256_short_v1.common import verify
    count=5 if mode=='short5' else 1
    if mode not in ('short1','short5') or len(folders)!=count:raise ValueError('Incomplete PA pair series')
    result=dict(schema='digit-unified-results-v2',dataset='PA',mode=mode,passed=True,complete=True,
        evidence_scope='native',source_sha256=verify(),row_bytes=512,feature_kind='real',label_kind='real',
        root_order_comparison='GIDS_random_DiGiT_BFS256_different_short_window_examples',
        accuracy_evaluated=False,full_epoch_training=False,stable_speedup_established=False,epochs=[],arms={},paired_rounds=[],
        aggregation='median_of_five_paired_speedups' if count==5 else 'single_pair_ratio_of_total_training_time')
    reports={a:[] for a in ('gids','digit')}
    for rep,folder in enumerate(folders,1):
        state=read(folder/'status.json')
        order=['gids','digit'] if rep%2 else ['digit','gids']
        expected=['gpu_checks']+['smoke_'+a for a in order]+order
        if not state['passed'] or not state['complete'] or state['completed']!=expected:raise ValueError('Unaccepted PA stages')
        for name in expected:
            if read(folder/(name+'_exit.json'))['returncode']!=0:raise ValueError('PA exit failed')
        pair={a:read(folder/a/'report.json') for a in reports};accepted=aggregate(pair)
        result['paired_rounds'].append(dict(round=rep,gids_seconds=pair['gids']['measured_seconds'],digit_seconds=pair['digit']['measured_seconds'],speedup=accepted['speedup']))
        for arm,r in pair.items():
            reports[arm].append(r);windows=r['windows'][1:]
            c=raw_counts(r['training'],sum(w['group_sampled_edges'] for w in windows),sum(w['outer_sampled_edges'] for w in windows),arm)
            result['epochs'].append(dict(arm=arm,round=rep,epoch=1,seconds=r['measured_seconds'],warmup_seconds=r['windows'][0]['seconds'],
                training_wall_seconds=None,measured_batches=300,measured_roots=300*1024,counters=c,metrics=metrics(c)))
    for arm,rows in reports.items():
        measured={k:0 for k in FIELDS};whole=dict(measured)
        for r,e in zip(rows,[e for e in result['epochs'] if e['arm']==arm]):
            measured=add_counts(measured,e['counters']);w=r['windows']
            whole=add_counts(whole,raw_counts(r['complete_region'],sum(x['group_sampled_edges'] for x in w),sum(x['outer_sampled_edges'] for x in w),arm))
        times=[r['measured_seconds'] for r in rows];cache=rows[0]['cache']
        for r in rows:
            if any(r[k]!=rows[0][k] for k in ('initial_model_sha256','source_sha256','protocol')):
                raise ValueError('PA repeated configuration drift')
            if any(r['cache'][k]!=cache[k] for k in ('cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes')):
                raise ValueError('PA repeated capacity drift')
        result['arms'][arm]=dict(per_round_seconds=times,training_seconds=sum(times),mean_training_epoch_seconds=sum(times)/count,
            measured_counters=measured,including_warmup_counters=whole,weighted_metrics=metrics(measured),
            cpu_feature_bytes=cache['cpu_feature_bytes'],gpu_feature_bytes=cache['gpu_feature_cache_bytes'])
    result['speedup']=statistics.median(p['speedup'] for p in result['paired_rounds'])
    a=result['arms']['gids']['measured_counters']['primary_bytes'];b=result['arms']['digit']['measured_counters']['primary_bytes']
    result['ssd_primary_bytes_reduction_fraction']=1-b/a if a else None
    return result
