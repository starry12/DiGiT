"""Gate both smoke arms before full runs; validate complete native evidence."""
import math
from training.gat.common import *
from runtime.io.accounting import validate_region,summarize
from ae.pa_sage.full_validation import checkpoint_hash,io_check

def report_check(r,arm,smoke,binding,folder):
    p=cfg();counts={k:source_config()['splits'][k]['count'] for k in ('train','valid','test')}
    require(r['passed'] is True and r['source_only'] is False and r['smoke']==smoke and r['arm']==arm and r['seed']==0,'Wrong native report')
    require(r['candidate_sha256']==binding['candidate_sha256'] and r['protocol_sha256']==binding['protocol_sha256'] and r['prepared_sha256']==binding['prepared_sha256'],'Wrong source/data binding')
    require(r['graph_mapping']=='private_copy_on_write','Wrong graph memory mapping')
    require(r['graph']==p['graph'] and r['metadata_mode']==p['metadata_mode'],'Wrong graph or native mode')
    require(r['model_name']=='gat' and r['model_config']==p['model_config'] and r['model_semantics']==p['model_semantics'] and r['model_parameter_count']==p['model_parameter_count'],'Wrong GAT model')
    require(r['optimizer_effective']==p['optimizer']['kwargs'],'Wrong GAT optimizer')
    require(not r['raw_ssd_writes'] and r['all_epochs_metadata_and_cache_reused'],'Raw write or state recreation')
    require(r['cache_bytes']==p['gpu_cache_bytes'] and r['cpu_cache_rows']==p['cpu_cache_rows'],'Cache budget mismatch')
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
    require((groups>0) if arm=='digit_full' else groups==0,'Wrong grouped sampling path')
    require(r['io_accounting_training']==summarize([(e['training'],e['train_seconds']) for e in r['epochs']]),'I/O summary differs')
    require(checkpoint_hash(folder/'final_model.pt')==r['final_parameters_sha256']==r['epochs'][-1]['model_sha256'],'Checkpoint differs')
    if smoke:
        require(r['test'] is None and not (folder/'test_invocation.json').exists() and len(r['audits'])==4,'Unexpected smoke test or missing feature audits')
    else:
        evaluate(r['test'],'test',None);inv=read(folder/'test_invocation.json')
        require(inv['checkpoint_epoch']==20 and inv['arm']==arm and inv['model_sha256']==r['final_parameters_sha256'] and inv['checkpoint_sha256']==sha(folder/'final_model.pt'),'Final-test checkpoint binding differs')
    return True

def pair_check(reports,smoke):
    require(set(reports)=={'gids','digit_full'},'Missing pair')
    a,b=reports['gids'],reports['digit_full']
    require(all(r['model_name']=='gat' and r['model_config']==cfg()['model_config'] and r['optimizer_effective']==cfg()['optimizer']['kwargs'] for r in reports.values()),'Unpaired GAT protocol')
    require(len(a['epochs'])==len(b['epochs'])==(2 if smoke else 20),'Incomplete pair epochs')
    require(a['initial_parameters_sha256']==b['initial_parameters_sha256'] and a['initial_dgl_rng']==b['initial_dgl_rng'],'Unpaired initialization')
    for x,y in zip(a['epochs'],b['epochs']):
        require(x['root_sha256']==y['root_sha256'] and x['validation']['trace_manifest_sha256']==y['validation']['trace_manifest_sha256'],'Different root order/evaluation trace')
        if smoke:require(x['validation']['bindings']==y['validation']['bindings'],'Different validation features/labels')
    return dict(passed=True,training_speedup=a['training_seconds']/b['training_seconds'],
                gids_training_seconds=a['training_seconds'],digit_training_seconds=b['training_seconds'],
                test_accuracy=None if smoke else {k:r['test']['accuracy'] for k,r in reports.items()},
                io={k:r['io_accounting_training'] for k,r in reports.items()},
                scope='short native gate only' if smoke else 'single seed0, reconstructed bidirectional PA/GAT, fixed inherited g2 layout')
