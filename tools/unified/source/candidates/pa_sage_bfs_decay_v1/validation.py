"""Gate full training, paired evaluation and equal feature-cache capacity."""
import math
from .schedule import lr_for_epoch,validate_schedule
from candidates.pa_sage_bfs_decay_v1.common import *
from candidates.io_accounting_v1.accounting import validate_region,summarize
from ae.pa_sage.full_validation import checkpoint_hash,io_check as base_io_check

def io_check(value,arm):
    policy=cfg()['arms'][arm]
    base_io_check(value,'digit_full' if policy['mixed_io'] else 'gids')
    require(value['gpu']['policy']==policy['gpu_policy'],'Wrong native GPU replacement policy')
    require(value['mixed']['enabled']==policy['mixed_io'],'Wrong mixed-I/O path')
    if not policy['cpu_cache_rows']:require(value['feature']['cpu']==0,'CPU-cache access in disabled arm')

def report_check(r,arm,smoke,binding,folder):
    p=cfg();counts={k:source_config()['splits'][k]['count'] for k in ('train','valid','test')}
    require(r['passed'] is True and r['source_only'] is False and r['smoke']==smoke and r['arm']==arm and r['seed']==0,'Wrong native report')
    require(r['candidate_sha256']==binding['candidate_sha256'] and r['protocol_sha256']==binding['protocol_sha256'] and r['prepared_sha256']==binding['prepared_sha256'],'Wrong source/data binding')
    require(r['graph_mapping']=='private_copy_on_write','Wrong graph memory mapping')
    require(r['graph']==p['graph'] and r['metadata_mode']==p['metadata_mode'],'Wrong graph or native mode')
    validate_schedule(r,arm,p,smoke)
    require(not r['raw_ssd_writes'] and r['all_epochs_metadata_and_cache_reused'],'Raw write or state recreation')
    from .cache import validate_binding
    validate_binding(r['cpu_binding'],arm,p,binding)
    require(r['cache_bytes']==p['gpu_cache_bytes'] and r['cpu_cache_rows']==p['arms'][arm]['cpu_cache_rows'] and r['policy']==p['arms'][arm],'Cache budget mismatch')
    require(r['admission']['passed'] and r['resource_observations']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'Resource budget exceeded')
    require(r['external_monitor']['sample_count']>0 and r['external_monitor']['monitor_peak_rss_bytes']<=64*2**20 and r['external_monitor']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'External monitor failed')
    n=p['smoke_epochs'] if smoke else p['epochs'];updates=2 if smoke else math.ceil(counts['train']/p['batch_size'])
    require(len(r['epochs'])==n and r['updates']==n*updates,'Incomplete training')
    evaluation_check(r,arm,smoke,folder)
    orders=binding['orders'][arm];groups=0
    for i,e in enumerate(r['epochs']):
        require(e['epoch']==i+1 and e['updates']==updates and e['root_sha256']==orders[str(i)]['digest'],'Wrong epoch extent/order')
        require(len(e['losses'])==updates and all(math.isfinite(v) for v in e['losses']) and e['metadata_reused'] and e['cache_object_reused'],'Incomplete/nonfinite epoch')
        require(sum(s['output_nodes'] for s in e['shapes'])==(updates*p['batch_size'] if smoke else counts['train']),'Incomplete train split')
        require(e['training']['useful_io']['region_id']==2*i+1 and e['validation']['useful_io']['region_id']==2*i+2,'Counter phase leakage')
        io_check(e['training'],arm);validate_region(e['training'])
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
            math.isclose(r['validation_seconds'],sum(e['validation']['seconds'] for e in r['epochs'])),'Timing totals differ')
    require(checkpoint_hash(folder/'final_model.pt')==r['final_parameters_sha256']==r['epochs'][-1]['model_sha256'],'Checkpoint differs')
    if smoke:require(len(r['audits'])==4,'Missing smoke feature audits')
    return True


def evaluation_check(r,arm,smoke,folder):
    p=cfg();n=p['smoke_epochs'] if smoke else p['epochs']
    lifecycle=r['evaluation_lifecycle']
    require(lifecycle['validation_calls']==n and lifecycle['test_calls']==(0 if smoke else 1)
            and lifecycle['diagnostic_replays']==0 and lifecycle['training_rng_unchanged'], 'Wrong evaluation lifecycle')
    from ae.pa_sage.full_validation import evaluation_check as base_evaluation_check
    contract=dict(batch_size=p['batch_size'],classes=p['classes'])
    for split,key in [('valid','validation_trace'),('test','test_trace')]:
        contract[split+'_examples']=2*p['batch_size'] if smoke else source_config()['splits'][split]['count']
        contract[split+'_trace_sha256']=sha(ROOT/p[key]/'manifest.json')
    for epoch in r['epochs']:
        value=epoch['validation']
        base_evaluation_check(value,'valid',contract,'digit_full' if p['arms'][arm]['mixed_io'] else 'gids')
        io_check(value,arm);validate_region(value)
    require(r['io_accounting_validation']==summarize([(e['validation'],e['validation']['seconds']) for e in r['epochs']]), 'Validation I/O summary differs')
    require(math.isclose(lifecycle['validation_seconds'],r['validation_seconds']), 'Validation timing differs')
    if smoke:
        require(r['test'] is None and r['io_accounting_test'] is None and not (folder/'test_invocation.json').exists(), 'Unexpected smoke test')
    else:
        value=r['test']
        base_evaluation_check(value,'test',contract,'digit_full' if p['arms'][arm]['mixed_io'] else 'gids')
        io_check(value,arm);validate_region(value)
        require(value['useful_io']['region_id']==2*n+1,'Final-test counter phase leakage')
        require(r['io_accounting_test']==summarize([(value,value['seconds'])]), 'Test I/O summary differs')
        require(math.isclose(lifecycle['test_seconds'],value['seconds']), 'Test timing differs')
        inv=read(folder/'test_invocation.json')
        require(inv['checkpoint_epoch']==n and inv['arm']==arm and inv['seed']==p['seed']
                and inv['model_sha256']==r['final_parameters_sha256']
                and inv['checkpoint_sha256']==sha(folder/'final_model.pt'), 'Final-test checkpoint binding differs')


def matrix_check(reports,smoke):
    p=cfg();n=p['smoke_epochs'] if smoke else p['epochs']
    require(set(reports)==set(p['ablation_order']),'Missing factorial arm')
    reference=reports['digit_random_constant']
    rows={}
    for arm,r in reports.items():
        policy=p['arms'][arm];schedule=policy['lr_schedule']
        a=reports['digit_random_'+schedule]
        require(r['initial_parameters_sha256']==reference['initial_parameters_sha256'] and
                r['initial_dgl_rng']==reference['initial_dgl_rng'],'Unpaired initialization')
        require(r['cpu_binding']==reference['cpu_binding'],'Hot set or cache differs')
        require(len(r['epochs'])==n,'Incomplete epochs')
        validate_schedule(r,arm,p,smoke)
        for x,y in zip(reference['epochs'],r['epochs']):
            require(x['validation']['trace_manifest_sha256']==y['validation']['trace_manifest_sha256'],'Different validation traces')
            if smoke:require(x['validation']['bindings']==y['validation']['bindings'],'Different validation features')
        if not smoke:require(reference['test']['trace_manifest_sha256']==r['test']['trace_manifest_sha256'],'Different test traces')
        cpu=sum(e['training']['feature']['cpu'] for e in r['epochs'])
        requests=sum(e['training']['feature']['cpu']+e['training']['feature']['gpu_ssd'] for e in r['epochs'])
        rows[arm]=dict(training_seconds=r['training_seconds'],mean_training_epoch_seconds=r['training_seconds']/n,
            epoch_training_seconds=[e['train_seconds'] for e in r['epochs']],
            learning_rates=[e['learning_rate'] for e in r['epochs']],
            validation_accuracy=[e['validation']['accuracy'] for e in r['epochs']],
            epoch20_validation_accuracy=None if smoke else r['epochs'][-1]['validation']['accuracy'],
            test_accuracy=None if smoke else r['test']['accuracy'],validation_seconds=r['validation_seconds'],
            test_seconds=None if smoke else r['test']['seconds'],setup_seconds=r['setup_seconds'],worker_seconds=r['worker_seconds'],
            trace_preparation_seconds=r['evaluation_lifecycle']['trace_preparation_seconds'],
            cpu_service_fraction=cpu/requests,feature_requests=requests,cpu_cache=r['cpu_binding'],io=r['io_accounting_training'],
            comparison_reference='digit_random_'+schedule,speedup_vs_random=a['training_seconds']/r['training_seconds'],
            validation_accuracy_delta_percentage_points=None if smoke else 100*(r['epochs'][-1]['validation']['accuracy']-a['epochs'][-1]['validation']['accuracy']),
            test_accuracy_delta_percentage_points=None if smoke else 100*(r['test']['accuracy']-a['test']['accuracy']))
    for epoch in range(n):
        for order in ('random','mix64','mix16'):
            a=reports['digit_'+order+'_constant']['epochs'][epoch]
            b=reports['digit_'+order+'_decay']['epochs'][epoch]
            require(a['root_sha256']==b['root_sha256'],'LR comparison changed root order')
        for schedule in ('constant','decay'):
            require(len({reports['digit_'+order+'_'+schedule]['epochs'][epoch]['root_sha256']
                         for order in ('random','mix64','mix16')})==3,'Order controls identical')
    effects={}
    for order in ('random','mix64','mix16'):
        a=rows['digit_'+order+'_constant'];b=rows['digit_'+order+'_decay']
        effects[order]=dict(training_speedup_constant_over_decay=a['training_seconds']/b['training_seconds'],
            validation_delta_percentage_points=None if smoke else 100*(b['epoch20_validation_accuracy']-a['epoch20_validation_accuracy']),
            test_delta_percentage_points=None if smoke else 100*(b['test_accuracy']-a['test_accuracy']))
    return dict(passed=True,smoke=smoke,rows=rows,decay_effect_by_order=effects,
        scope='Smoke correctness only' if smoke else p['scope'],
        limits=['Single seed, no statistical equivalence claim','All six arms rerun; shared graph, initialization, cache and evaluation traces',
                'Predetermined schedule, no validation/test-driven adaptation','Training time excludes preparation, validation and test'])
