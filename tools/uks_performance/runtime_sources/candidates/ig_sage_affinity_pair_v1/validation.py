import math
from candidates.ig_sage_affinity_pair_v1.common import *
from candidates.ig_sage_affinity_pair_v1.windows import root_slices,check_windows
from candidates.io_accounting_v1.accounting import summarize,validate_region

def report_check(r,arm,smoke,binding,folder):
    import torch
    p=cfg();ranges=root_slices(p,smoke)
    require(r['schema']=='digit-ig-window-report-v1' and r['passed'] and r['arm']==arm and r['smoke']==smoke,'Wrong IG report')
    require(r['candidate_sha256']==binding['candidate_sha256'] and r['protocol_sha256']==binding['protocol_sha256'],'Wrong version')
    require(r['model_name'] in p['models'] and r['seed']==0 and r['epochs'] is None,'Wrong model/extent')
    from candidates.ig_sage_affinity_pair_v1.model import model_config,make_model
    require(r['model_config']==model_config(r['model_name']) and r['optimizer']==p['optimizer'],'Model/optimizer changed')
    require(r['model_parameter_count']==sum(v.numel() for v in make_model(r['model_name']).parameters()),'Model size differs')
    require(r['validation'] is None and r['validation_calls']==0 and r['test'] is None and r['test_calls']==0 and r['accuracy'] is None and not r['final_accuracy_claim'] and not r['epoch_time_claim'] and not r['steady_state_proven'] and r['diagnostic_replays']==0,'Unexpected evaluation/epoch/accuracy scope')
    require(not r['raw_ssd_writes'] and r['all_metadata_and_cache_reused'] and r['route_warmup']==smoke,'Runtime protocol changed')
    require(r['updates']==(p['smoke_train_batches'] if smoke else p['total_train_batches']) and r['measured_batches']==(0 if smoke else cfg()['measured_batches']),'Incomplete bounded run')
    require((r['warmup'] is None)==smoke,'Wrong warmup scope')
    prepared=binding['prepared_orders']
    for name,lo,hi,width in ranges:
        count=(hi-lo)*p['batch_size'];v=r[name]
        require(v['examples']==count and v['batches']==hi-lo and len(v['losses'])==hi-lo,'Incomplete phase '+name)
        require(all(math.isfinite(x) for x in v['losses']) and math.isfinite(v['seconds']) and v['seconds']>0,'Invalid loss/time')
        require(v['phase_wall_seconds']>=v['seconds'] and r[name+'_seconds']==v['seconds'],'Wrong phase timing')
        check_windows(v['windows'],v['batches'],width,v['seconds'])
        require(v['window_count']==len(v['windows']),'Wrong window count')
        require(v['sampling_shape_totals'][0][0]==v['feature_rows'] and v['sampling_shape_totals'][-1][1]==count,'Wrong sampled extent')
        validate_region(v);require(r['io_accounting_'+name]==summarize([(v,v['seconds'])]),'I/O totals differ')
        require(read(folder/(name+'.json'))==v,'Phase report changed')
        key='smoke_payload_sha256' if smoke else name+'_payload_sha256'
        require(v['roots_sha256']==prepared[key],'Wrong frozen roots')
        if name=='training' and not smoke:require([w['roots_sha256'] for w in v['windows']]==prepared['training_window_payload_sha256'],'Wrong per-window roots')
        require(v['group_edges']==sum(w['group_edges'] for w in v['windows']) and v['outer_edges']==sum(w['outer_edges'] for w in v['windows']),'Group totals differ')
        require(0<=v['group_edges']<=v['outer_edges'],'Invalid group coverage')
        require(v['group_edges']>0 if arm=='digit_full' else v['group_edges']==0,'Wrong grouped sampling path')
    require(r['training']['useful_io']['region_id']==2,'Wrong training counter region')
    if not smoke:require(r['warmup']['useful_io']['region_id']==1,'Warmup counter leakage')
    require(sha(folder/'final_model.pt')==r['checkpoint_sha256'],'Checkpoint file changed')
    checkpoint=torch.load(folder/'final_model.pt',map_location='cpu');require(checkpoint['epoch'] is None and checkpoint['updates']==r['updates'] and checkpoint['model_name']==r['model_name'],'Wrong checkpoint')
    require(state_hash(checkpoint['model'])==r['final_parameters_sha256'] and r['initial_parameters_sha256']!=r['final_parameters_sha256'],'Checkpoint/model update mismatch')
    states=checkpoint['optimizer']['state'];require(states and all(int(v['step'])==r['updates'] for v in states.values()),'Optimizer update count differs')
    require(r['native']['route_warmup']==smoke,'Warmup receipt differs')
    require(r['native']['cache_rows']==p['cpu_cache_rows'] and r['native']['native_info']['gpu_cache_bytes']==p['gpu_cache_bytes'],'Cache allocation differs')
    if smoke:require(r['routing']['passed'] and len(r['source_audits'])==p['smoke_train_batches'] and all(a['source_equal'] for a in r['source_audits']),'Missing native correctness audits')
    else:require(not r['source_audits'] and r['routing'] is None,'Audit overhead on formal path')
    require(r['admission']['passed'] and r['external_monitor']['observed_peak_device_used_bytes']<=r['admission']['required_bytes'] and r['peak_host_rss_kib']*1024<=r['admission']['host_required_bytes'],'Resource budget exceeded')
    return True

def pair_check(reports,smoke):
    require(set(reports)=={'gids','digit_full'},'Missing pair');a,b=[reports[x] for x in ('gids','digit_full')]
    for k in ('model_name','model_config','optimizer','initial_parameters_sha256','seed','epochs','smoke','updates','measured_batches'):require(a[k]==b[k],'Unpaired '+k)
    require(a['smoke']==smoke,'Wrong pair scope')
    for name,lo,hi,width in root_slices(cfg(),smoke):
        require(a[name]['roots_sha256']==b[name]['roots_sha256'] and a[name]['examples']==b[name]['examples'],'Unpaired roots')
        require([w['roots_sha256'] for w in a[name]['windows']]==[w['roots_sha256'] for w in b[name]['windows']],'Unpaired window roots')
    return dict(passed=True,scope='source/native correctness smoke only' if smoke else cfg()['scope'],model=a['model_name'],epochs=None,warmup_batches=0 if smoke else cfg()['warmup_batches'],measured_batches=0 if smoke else cfg()['measured_batches'],measurement_windows=0 if smoke else cfg()['measurement_windows'],
                training_speedup=None if smoke else a['training_seconds']/b['training_seconds'],test_accuracy=None,final_accuracy_claim=False,epoch_time_claim=False,steady_state_proven=False)
