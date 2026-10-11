"""Gate both smoke arms before full runs; validate complete native evidence."""
import math
from candidates.pa_sage_ablation_v1.common import *
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
    require(r['optimizer_effective']==p['optimizer']['kwargs'],'Wrong selected optimizer')
    require(not r['raw_ssd_writes'] and r['all_epochs_metadata_and_cache_reused'],'Raw write or state recreation')
    require(r['cache_bytes']==p['gpu_cache_bytes'] and r['cpu_cache_rows']==p['arms'][arm]['cpu_cache_rows'] and r['policy']==p['arms'][arm],'Cache budget mismatch')
    require(r['admission']['passed'] and r['resource_observations']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'Resource budget exceeded')
    require(r['external_monitor']['sample_count']>0 and r['external_monitor']['monitor_peak_rss_bytes']<=64*2**20 and r['external_monitor']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'External monitor failed')
    n=2 if smoke else 20;updates=2 if smoke else math.ceil(counts['train']/p['batch_size'])
    require(len(r['epochs'])==n and r['updates']==n*updates,'Incomplete training')
    lifecycle=r['evaluation_lifecycle'];require(lifecycle['validation_calls']==n and lifecycle['test_calls']==(0 if smoke else 1) and lifecycle['diagnostic_replays']==0 and lifecycle['training_rng_unchanged'],'Wrong evaluation lifecycle')
    def evaluate(v,split,limit):
        expected=limit*p['batch_size'] if limit else counts[split]
        require(v['examples']==expected and v['batches']==math.ceil(expected/p['batch_size']),'Incomplete '+split)
        require(v['training_rng_restored'] and math.isfinite(v['loss']) and 0<=v['accuracy']<=1,'Bad evaluation')
        require(sum(v['prediction_histogram'])==expected and math.isclose(sum(v['per_class_correct'])/expected,v['accuracy'],abs_tol=1e-12),'Evaluation counts mismatch')
        path=ROOT/p['validation_trace' if split=='valid' else 'test_trace'];require(v['trace_manifest_sha256']==sha(path/'manifest.json'),'Wrong selected trace')
        io_check(v,arm);validate_region(v)
    orders=read(ROOT/p['base_layout']/'orders.json')['hashes'];groups=0
    for i,e in enumerate(r['epochs']):
        require(e['epoch']==i+1 and e['updates']==updates and e['root_sha256']==orders['s0_e%d'%i],'Wrong epoch extent/order')
        require(len(e['losses'])==updates and all(math.isfinite(v) for v in e['losses']) and e['metadata_reused'] and e['cache_object_reused'],'Incomplete/nonfinite epoch')
        require(sum(s['output_nodes'] for s in e['shapes'])==(updates*p['batch_size'] if smoke else counts['train']),'Incomplete train split')
        require(e['training']['useful_io']['region_id']==2*i+1 and e['validation']['useful_io']['region_id']==2*i+2,'Counter phase leakage')
        io_check(e['training'],arm);validate_region(e['training']);evaluate(e['validation'],'valid',2 if smoke else None)
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
        if arm in ('gids','gr') and not smoke:require(all(e['selected_source_bridge'].values()),'Selected-source trajectory bridge failed')
    require((groups>0) if p['arms'][arm]['group_sampling'] else groups==0,'Wrong grouped sampling path')
    require(r['io_accounting_training']==summarize([(e['training'],e['train_seconds']) for e in r['epochs']]),'I/O summary differs')
    require(math.isclose(r['training_seconds'],sum(e['train_seconds'] for e in r['epochs'])) and
            math.isclose(r['validation_seconds'],sum(e['validation']['seconds'] for e in r['epochs'])),'Timing totals differ')
    require(checkpoint_hash(folder/'final_model.pt')==r['final_parameters_sha256']==r['epochs'][-1]['model_sha256'],'Checkpoint differs')
    if smoke:
        require(r['test'] is None and not (folder/'test_invocation.json').exists() and len(r['audits'])==4,'Unexpected smoke test or missing feature audits')
    else:
        evaluate(r['test'],'test',None);inv=read(folder/'test_invocation.json')
        require(inv['checkpoint_epoch']==20 and inv['arm']==arm and inv['model_sha256']==r['final_parameters_sha256'] and inv['checkpoint_sha256']==sha(folder/'final_model.pt'),'Final-test checkpoint binding differs')
    return True

def matrix_check(reports,smoke):
    p=cfg()
    require(set(reports)==set(p['ablation_order']),'Missing ablation arm')
    baseline=reports['gids']
    for arm,r in reports.items():
        require(r['initial_parameters_sha256']==baseline['initial_parameters_sha256'] and
                r['initial_dgl_rng']==baseline['initial_dgl_rng'],'Unpaired initialization')
        require(len(r['epochs'])==len(baseline['epochs']),'Different epoch count')
        for x,y in zip(baseline['epochs'],r['epochs']):
            require(x['root_sha256']==y['root_sha256'] and
                    x['validation']['trace_manifest_sha256']==y['validation']['trace_manifest_sha256'],
                    'Different root order/evaluation trace')
            if smoke:require(x['validation']['bindings']==y['validation']['bindings'],'Different evaluation features/labels')
    # Changing physical layout/cache must never change the learned trajectory.
    for left,right in (('gids','gr'),('ns','digit_full')):
        a,b=reports[left],reports[right]
        for x,y in zip(a['epochs'],b['epochs']):
            for field in ('losses','model_sha256','shapes'):
                require(x[field]==y[field],'Trajectory changed: '+left+'/'+right+' '+field)
            require(x['validation']['prediction_sha256']==y['validation']['prediction_sha256'],
                    'Validation predictions changed within sampler pair')
        if smoke:
            def logical(audit):
                return {k:v for k,v in audit.items() if k not in ('storage_rows','storage_flags')}
            require([logical(a) for a in a['audits']]==[logical(a) for a in b['audits']],
                    'Native feature/block/gradient trajectory parity failed')
        else:
            require(a['test']['prediction_sha256']==b['test']['prediction_sha256'],
                    'Final test predictions changed within sampler pair')
    return dict(passed=True,smoke=smoke,
        scope='Native smoke only; not performance evidence' if smoke else
              'PA/SAGE four cumulative stages, seed0 once, 20 epochs each; reconstructed no-BFS g2 layout',
        matched_sampler_pairs=[['gids','gr'],['ns','digit_full']],
        arms={arm:dict(label=p['arms'][arm]['label'],policy=p['arms'][arm],
            training_seconds=r['training_seconds'],mean_training_epoch_seconds=r['training_seconds']/len(r['epochs']),
            order_excluded_training_seconds=sum(e['train_seconds']-e['order_seconds'] for e in r['epochs']),
            validation_seconds=r['validation_seconds'],
            setup_seconds=r.get('setup_seconds'),metadata_setup_seconds=r.get('metadata_setup_seconds'),
            trace_preparation_seconds=r.get('evaluation_lifecycle',{}).get('trace_preparation_seconds'),
            test_seconds=None if smoke else r['test']['seconds'],
            test_accuracy=None if smoke else r['test']['accuracy'],
            speedup_vs_gids=baseline['training_seconds']/r['training_seconds'],
            group_edges=sum(w['group_edges'] for e in r['epochs'] for w in e.get('windows',[])),
            target_edges=sum(w['target_edges'] for e in r['epochs'] for w in e.get('windows',[])),
            shortfall_edges=sum(w['shortfall_edges'] for e in r['epochs'] for w in e.get('windows',[])),
            cpu_feature_rows=sum(e.get('training',{}).get('feature',{}).get('cpu',0) for e in r['epochs']),
            gpu_hits=sum(e.get('training',{}).get('gpu',{}).get('hits',0) for e in r['epochs']),
            gpu_requests=sum(e.get('training',{}).get('gpu',{}).get('requests',0) for e in r['epochs']),
            io=r['io_accounting_training']) for arm,r in reports.items()})
