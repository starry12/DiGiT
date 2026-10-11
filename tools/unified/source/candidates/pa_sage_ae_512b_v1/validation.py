"""Gate all smoke arms before one-epoch performance runs."""
import math
from candidates.pa_sage_ae_512b_v1.common import *
from candidates.io_accounting_v1.accounting import validate_region,summarize
from ae.pa_sage.full_validation import checkpoint_hash

def io_check(value,arm):
    policy=cfg()['arms'][arm]
    device=value['device'];gpu=value['gpu']
    require(device['enabled'] and device['reconciled'] and device['submitted_commands']==device['completed_commands'],'Unfinished device I/O')
    require(gpu['capacity_pages']==cfg()['gpu_cache_bytes']//512 and gpu['reconciled'],'Wrong cache capacity')
    require(gpu['requests']==value['feature']['gpu_ssd'] and gpu['policy']==policy['gpu_policy'],'Wrong cache routing/policy')
    require(value['mixed']['enabled']==policy['mixed_io'] and value['reconciliation']['device_counters_checked'],'Missing command reconciliation')
    from candidates.pa_sage_mixed_group_v1.common import check_pair_counts
    check_pair_counts(value['pair_counts'],device,policy['mixed_io'],require_pair=False)
    if not policy['cpu_cache_rows']:require(value['feature']['cpu']==0,'CPU-cache access in disabled arm')

def report_check(r,arm,smoke,binding,folder):
    p=cfg();counts={'train':source_config()['splits']['train']['count']}
    require(r['passed'] is True and r['source_only'] is False and r['smoke']==smoke and r['arm']==arm and r['seed']==0,'Wrong native report')
    require(r['candidate_sha256']==binding['candidate_sha256'] and r['protocol_sha256']==binding['protocol_sha256'] and r['prepared_sha256']==binding['prepared_sha256'],'Wrong source/data binding')
    require(r['graph_mapping']=='anonymous_copied','Wrong graph memory mapping')
    require(r['graph']==p['graph'] and r['metadata_mode']==p['metadata_mode'],'Wrong graph or native mode')
    require(r['page_bytes']==512 and r['read_schedule']=='synchronous_single_batch','Wrong read granularity/schedule')
    require(r['optimizer_effective']==p['optimizer']['kwargs'],'Wrong selected optimizer')
    require(not r['raw_ssd_writes'] and r['all_epochs_metadata_and_cache_reused'],'Raw write or state recreation')
    from .cache import validate_binding
    validate_binding(r['cpu_binding'],arm,p,binding)
    require(r['cache_bytes']==p['gpu_cache_bytes'] and r['cpu_cache_rows']==p['arms'][arm]['cpu_cache_rows'] and r['policy']==p['arms'][arm],'Cache budget mismatch')
    require(r['admission']['passed'] and r['resource_observations']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'Resource budget exceeded')
    require(r['external_monitor']['sample_count']>0 and r['external_monitor']['monitor_peak_rss_bytes']<=64*2**20 and r['external_monitor']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'External monitor failed')
    n=2 if smoke else 1;updates=2 if smoke else math.ceil(counts['train']/p['batch_size'])
    require(len(r['epochs'])==n and r['updates']==n*updates,'Incomplete training')
    no_evaluation_check(r,folder)
    orders=read(ROOT/p['base_layout']/'orders.json')['hashes'];groups=0
    for i,e in enumerate(r['epochs']):
        require(e['epoch']==i+1 and e['updates']==updates and e['root_sha256']==orders['s0_e%d'%i],'Wrong epoch extent/order')
        require(len(e['losses'])==updates and all(math.isfinite(v) for v in e['losses']) and e['metadata_reused'] and e['cache_object_reused'],'Incomplete/nonfinite epoch')
        require(sum(s['output_nodes'] for s in e['shapes'])==(updates*p['batch_size'] if smoke else counts['train']),'Incomplete train split')
        require(e['training']['useful_io']['region_id']==i+1,'Counter phase leakage')
        io_check(e['training'],arm);validate_region(e['training'])
        from candidates.pa_sage_mixed_group_v1.common import check_pair_counts
        check_pair_counts(e['training']['pair_counts'],e['training']['device'],
                          p['arms'][arm]['mixed_io'],require_pair=not smoke)
        require(e['train_seconds']>0 and 0<=e['order_seconds']<=e['train_seconds'],'Bad epoch timing')
        groups+=sum(w['group_edges'] for w in e['windows'])
        require(sum(w['input_nodes'] for w in e['windows'])==sum(s['input_nodes'] for s in e['shapes']),'Window feature count differs')
        cursor=1
        for window in e['windows']:
            require(window['start_update']==cursor and cursor<=window['end_update']<=updates,'Gapped/duplicate training window')
            cursor=window['end_update']+1
            require(0<=window['actual_edges']<=window['target_edges'] and window['shortfall_edges']==window['target_edges']-window['actual_edges'],'Invalid sampling coverage')
            io_check(window,arm)
        require(cursor==updates+1,'Incomplete windows')
    require((groups>0) if p['arms'][arm]['group_sampling'] else groups==0,'Wrong grouped sampling path')
    require(r['io_accounting_training']==summarize([(e['training'],e['train_seconds']) for e in r['epochs']]),'I/O summary differs')
    require(math.isclose(r['training_seconds'],sum(e['train_seconds'] for e in r['epochs'])) and
            r['validation_seconds']==0,'Timing totals differ')
    require(checkpoint_hash(folder/'final_model.pt')==r['final_parameters_sha256']==r['epochs'][-1]['model_sha256'],'Checkpoint differs')
    if smoke:require(len(r['audits'])==4,'Missing smoke feature audits')
    return True

def no_evaluation_check(r,folder):
    lifecycle=r['evaluation_lifecycle']
    require(lifecycle['enabled'] is False and lifecycle['validation_calls']==0 and lifecycle['test_calls']==0 and lifecycle['diagnostic_replays']==0 and lifecycle['training_rng_unchanged'] and lifecycle['trace_preparation_seconds']==0,'Unexpected evaluation lifecycle')
    require(r['test'] is None and r['validation_seconds']==0 and r['io_accounting_validation'] is None and r['io_accounting_test'] is None,'Unexpected evaluation result')
    require(all(e['validation'] is None for e in r['epochs']) and not (folder/'test_invocation.json').exists(),'Unexpected validation/test execution')


def matrix_check(reports, smoke):
    p = cfg()
    require(set(reports) == set(p['ablation_order']), 'Missing comparison arm')
    baseline = reports['gids']
    for arm, r in reports.items():
        require(r['initial_parameters_sha256'] == baseline['initial_parameters_sha256'] and
                r['initial_dgl_rng'] == baseline['initial_dgl_rng'], 'Initialization differs')
        require(len(r['epochs']) == len(baseline['epochs']), 'Epoch counts differ')
        for a, b in zip(baseline['epochs'], r['epochs']):
            require(a['root_sha256'] == b['root_sha256'], 'Root order differs')
            if arm != 'digit_full':
                for key in ('losses', 'loss_sha256', 'model_sha256', 'shapes'):
                    require(a[key] == b[key], 'GIDS trajectory differs: ' + arm + '/' + key)
        if smoke and arm != 'digit_full':
            require(r['audits'] == baseline['audits'], 'GIDS feature/block/gradient audit differs')
    rows = {}
    for arm, r in reports.items():
        cpu = sum(e['training']['feature']['cpu'] for e in r['epochs'])
        requests = sum(e['training']['feature']['cpu'] + e['training']['feature']['gpu_ssd']
                       for e in r['epochs'])
        rows[arm] = dict(training_seconds=r['training_seconds'], setup_seconds=r['setup_seconds'],
                         cpu_service_fraction=cpu / requests, feature_requests=requests,
                         cpu_cache=r['cpu_binding'], io=r['io_accounting_training'],
                         epochs=[epoch_metrics(e) for e in r['epochs']],
                         speedup_vs_gids=baseline['training_seconds'] / r['training_seconds'])
    require(all(row['cpu_service_fraction']>0 for row in rows.values()),'CPU cache did not serve features')
    return dict(passed=True, smoke=smoke, rows=rows,
                digit_speedup_vs_gids=baseline['training_seconds']/reports['digit_full']['training_seconds'],
                scope='Smoke correctness only' if smoke else p['scope'],
                limits=['Single first epoch, seed 0; not a 20-epoch or accuracy acceptance',
                        'Original AE CPU row sets retained with explicit lookup; GPU payload capacity unchanged',
                        'Same pair-capable binary, single-batch blocking reads; no async compute overlap'])


def epoch_metrics(e):
    x=e['training'];f=x['feature'];g=x['gpu'];d=x['device'];total=f['cpu']+f['gpu_ssd'];counts=x['pair_counts']
    return dict(epoch=e['epoch'],seconds=e['train_seconds'],logical_requests=total,
        cpu_service_fraction=f['cpu']/total,gpu_service_fraction=g['hits']/total,ssd_service_fraction=g['inserts']/total,
        gpu_conditional_hit_ratio=g['hit_rate'],ssd_primary_commands=d['primary_commands'],ssd_primary_bytes=d['primary_bytes'],
        ssd_completed_commands=d['completed_commands'],ssd_completed_bytes=d['completed_bytes'],ssd_active_seconds=d['active_ns']/1e9,
        primary_512b_commands=counts[0],primary_1k_commands=counts[1],ssd_grouped_row_fraction=2*counts[1]/g['inserts'] if g['inserts'] else 0,
        group_sampled_edges=sum(w['group_edges'] for w in e['windows']),sampled_edges=sum(w['actual_edges'] for w in e['windows']))
