"""Gate each point smoke before its one-epoch performance run."""
import math
from candidates.pa_sage_layout_shared_resume_v3.common import *
from candidates.io_accounting_v1.accounting import validate_region,summarize
from ae.pa_sage.full_validation import checkpoint_hash,io_check as base_io_check

def io_check(value,arm):
    policy=cfg()['arms'][arm]
    base_io_check(value,'digit_full' if policy['mixed_io'] else 'gids')
    require(value['gpu']['policy']==policy['gpu_policy'],'Wrong native GPU replacement policy')
    require(value['mixed']['enabled']==policy['mixed_io'],'Wrong mixed-I/O path')
    if not policy['cpu_cache_rows']:require(value['feature']['cpu']==0,'CPU-cache access in disabled arm')

def report_check(r,arm,smoke,binding,folder):
    require(r['verification_policy']==verification_policy(r['point']),'Wrong verification scope')
    require(not smoke or r['verification_policy']['independent_native_smoke'],'Unexpected independent smoke')
    p=cfg();counts={'train':source_config()['splits']['train']['count']}
    require(r['feature_mode']==p['feature_mode'] and r['pool_receipt_sha256']==binding['pool_receipt_sha256'],'Wrong feature source')
    require(r['point']==binding['point']==p['point'] and r['layout_manifest_sha256']==binding['layout_manifest_sha256'],'Wrong point/layout report')
    require(r['overlay_receipt_sha256']==binding['prepared_sha256'],'Wrong point overlay')
    require(r['passed'] is True and r['source_only'] is False and r['smoke']==smoke and r['arm']==arm and r['seed']==0,'Wrong native report')
    require(r['candidate_sha256']==binding['candidate_sha256'] and r['protocol_sha256']==binding['protocol_sha256'] and r['prepared_sha256']==binding['prepared_sha256'],'Wrong source/data binding')
    require(r['graph_mapping']=='private_copy_on_write','Wrong graph memory mapping')
    require(r['graph']==p['graph'] and r['metadata_mode']==p['metadata_mode'],'Wrong graph or native mode')
    require(r['optimizer_effective']==p['optimizer']['kwargs'],'Wrong selected optimizer')
    require(not r['raw_ssd_writes'] and r['all_epochs_metadata_and_cache_reused'],'Raw write or state recreation')
    require(r['cache_bytes']==p['gpu_cache_bytes'] and r['cpu_cache_rows']==p['arms'][arm]['cpu_cache_rows'] and r['policy']==p['arms'][arm],'Cache budget mismatch')
    require(r['admission']['passed'] and r['resource_observations']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'Resource budget exceeded')
    require(r['external_monitor']['sample_count']>0 and r['external_monitor']['monitor_peak_rss_bytes']<=64*2**20 and r['external_monitor']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'],'External monitor failed')
    n=2 if smoke else 1;updates=2 if smoke else math.ceil(counts['train']/p['batch_size'])
    require(len(r['epochs'])==n and r['updates']==n*updates,'Incomplete training')
    no_evaluation_check(r,folder)
    orders=read(p['orders_file'])['hashes'];groups=0
    for i,e in enumerate(r['epochs']):
        require(e['epoch']==i+1 and e['updates']==updates and e['root_sha256']==orders['s0_e%d'%i],'Wrong epoch extent/order')
        require(len(e['losses'])==updates and all(math.isfinite(v) for v in e['losses']) and e['metadata_reused'] and e['cache_object_reused'],'Incomplete/nonfinite epoch')
        require(sum(s['output_nodes'] for s in e['shapes'])==(updates*p['batch_size'] if smoke else counts['train']),'Incomplete train split')
        require(e['training']['useful_io']['region_id']==i+1,'Counter phase leakage')
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
            r['validation_seconds']==0,'Timing totals differ')
    require(checkpoint_hash(folder/'final_model.pt')==r['final_parameters_sha256']==r['epochs'][-1]['model_sha256'],'Checkpoint differs')
    if smoke:require(len(r['audits'])==4,'Missing smoke feature audits')
    return True

def point_summary(reports,smoke):
    require(set(reports)=={'digit_full'},'Unexpected point arms')
    r=reports['digit_full'];p=cfg()
    return dict(passed=True,smoke=smoke,point=p['point'],protocol_sha256=sha(P),
        feature_mode=p['feature_mode'],pool_receipt_sha256=r['pool_receipt_sha256'],
        verification_policy=p['verification_policy'],
        layout_manifest_sha256=r['layout_manifest_sha256'],
        initial_parameters_sha256=r['initial_parameters_sha256'],initial_dgl_rng=r['initial_dgl_rng'],
        root_hashes=[e['root_sha256'] for e in r['epochs']],
        evaluation='disabled',
        training_seconds=r['training_seconds'],mean_training_epoch_seconds=r['training_seconds']/len(r['epochs']),
        order_excluded_training_seconds=sum(e['train_seconds']-e['order_seconds'] for e in r['epochs']),
        validation_seconds=r['validation_seconds'],test_seconds=None,
        test_accuracy=None,io=r['io_accounting_training'],
        group_edges=sum(w['group_edges'] for e in r['epochs'] for w in e['windows']),
        shortfall_edges=sum(w['shortfall_edges'] for e in r['epochs'] for w in e['windows']),
        scope='native smoke only' if smoke else 'one complete first seed0 epoch; no accuracy or steady-state claim')

def mode_pair_check(smoke,full):
    require(smoke['smoke'] is True and full['smoke'] is False,'Wrong point run modes')
    for key in ('point','protocol_sha256','layout_manifest_sha256','overlay_receipt_sha256','initial_parameters_sha256','initial_dgl_rng'):
        require(smoke[key]==full[key],'Smoke/full binding differs: '+key)
    for a,b in zip(smoke['epochs'],full['epochs']):
        require(a['root_sha256']==b['root_sha256'],'Smoke/full data order differs')


def no_evaluation_check(r,folder):
    lifecycle=r['evaluation_lifecycle']
    require(lifecycle['enabled'] is False and lifecycle['validation_calls']==0 and lifecycle['test_calls']==0 and lifecycle['diagnostic_replays']==0 and lifecycle['training_rng_unchanged'] and lifecycle['trace_preparation_seconds']==0,'Unexpected evaluation lifecycle')
    require(r['test'] is None and r['validation_seconds']==0 and r['io_accounting_validation'] is None and r['io_accounting_test'] is None,'Unexpected evaluation result')
    require(all(e['validation'] is None for e in r['epochs']) and not (folder/'test_invocation.json').exists(),'Unexpected validation/test execution')
