"""Revalidate and export bounded IG windows, without accuracy or epoch estimates."""
import argparse,csv,json,math,sys,statistics
from pathlib import Path
from candidates.ig_perf_v5.common import *
from candidates.ig_perf_v5.validation import pair_check
from candidates.pa_gat_native_v1.summarize import phase_metrics

def window_metrics(w):
    # A window may consume a companion row fetched in an earlier window.
    # Complete-region useful <= fill validation applies only after aggregating
    # all three windows, never independently to a partial counter interval.
    active=w['device_raw'][5]/1e9
    return dict(ssd_completed_bytes=w['device_bytes'],ssd_primary_bytes=w['primary_bytes'],ssd_replay_bytes=w['replay_bytes'],ssd_useful_consumed_bytes=w['useful']['ssd_useful_bytes'],ssd_active_seconds=active,
                ssd_physical_gbps=w['device_bytes']/active/1e9 if active else None,ssd_useful_gbps=None,
                useful_throughput_scope='See complete measured-region throughput; a window can consume fills from earlier windows',
                feature_cpu_rows=w['feature_cpu'],feature_gpu_ssd_rows=w['feature_gpu_ssd'],feature_seconds=w['feature_seconds'])

def metrics(arm,r):
    train=r['training'];f=train['feature'];seconds=[w['seconds'] for w in train['windows']]
    windows=[dict(index=w['index'],batches=w['batches'],seconds=w['seconds'],seconds_per_batch=w['seconds']/w['batches'],training_roots_per_second=w['batches']*cfg()['batch_size']/w['seconds'],roots_sha256=w['roots_sha256'],group_edge_share=w['group_edges']/w['outer_edges'],io=window_metrics(w)) for w in train['windows']]
    return dict(dataset='IG',model=r['model_name'],arm=arm,epochs=None,updates=r['updates'],warmup_batches=20,measured_batches=90,measurement_windows=3,training_seconds=r['training_seconds'],seconds_per_batch=r['training_seconds']/90,training_roots_per_second=90*cfg()['batch_size']/r['training_seconds'],warmup_seconds=r['warmup_seconds'],setup_seconds=r['setup_seconds'],worker_seconds=r['worker_seconds'],accuracy=None,validation_calls=0,test_calls=0,epoch_time_seconds=None,steady_state_proven=False,
                windows=windows,window_relative_spread=(max(seconds)-min(seconds))/statistics.median(seconds),
                outer_layer_group_edge_share=train['group_edges']/train['outer_edges'],cpu_feature_request_share=f['cpu']/(f['cpu']+f['gpu_ssd']),
                io={name:phase_metrics([(r[name],r[name]['seconds'])],name) for name in ('warmup','training')},observed_peak_gpu_bytes=r['external_monitor']['observed_peak_device_used_bytes'],monitor=r['external_monitor'],final_parameters_sha256=r['final_parameters_sha256'])
def comparison(reports,provenance,source):
    pair_check(reports,False);rows=[metrics(a,reports[a]) for a in ('gids','digit_full')];a,b=rows;strict=all(r['monitor']['strict_monitor_passed'] for r in rows)
    den=a['io']['training']['ssd_completed_bytes']
    return dict(schema='digit-ig-window-results-v1',passed=True,complete=True,training_performance_io_passed=True,strict_resource_acceptance=strict,qualification='complete' if strict else 'complete_with_monitoring_gaps',provenance=provenance,source=source,rows=rows,protocol_sha256=sha(P),optimizer=cfg()['optimizer'],
                ratios=dict(training_speedup=a['training_seconds']/b['training_seconds'],worker_speedup=a['worker_seconds']/b['worker_seconds'],training_ssd_bytes_reduction_percent=None if not den else 100*(1-b['io']['training']['ssd_completed_bytes']/den)),seed_count=1,independent_repeat_count=1,consecutive_window_count=3,final_accuracy_claim=False,epoch_time_claim=False,steady_state_proven=False,full_submission_matrix_complete=False,
                limitations=['Each arm performs 20 warmup training batches and three consecutive timed windows of 30 batches, seed0, batch1024. No validation or test. No accuracy, convergence or full-epoch timing evidence.', 'Three windows share one worker, model, optimizer and warmed cache; they are not independent repetitions and do not prove steady state. Window variation is reported without selecting favorable windows.', 'Timing uses CUDA-synchronized wall time for sampling, native feature reads/copies, forward, loss, backward and optimizer. Setup, source audits, warmup, counter snapshots and report writing are excluded and reported separately.', 'SSD GB/s uses SSD-active time; logical feature supply includes cache hits and uses native feature-call time. Per-window counters and aggregate bytes are retained.', 'Untuned paper reconstruction: inherited graph, g2 layout, cache policy and all three model/optimizer settings. GAT has four 128-wide hidden heads.', 'Sampled GPU peaks are lower bounds. Strict zero-error independent monitoring is required. CPU row selection and cache policy are system differences, not isolated sampler effects.'])

def load_formal(folder,allow_smoke=False):
    from candidates.ig_perf_v5.controller import plan,worker_command
    from candidates.ig_perf_v5.review import review_worker
    folder=Path(folder).resolve();state=read(folder/'status.json');launch=read(folder/'launch.json');binding=read(folder/'inputs.json')
    require(state['schema']=='digit-ig-window-run-v1' and state['passed'] and state['complete'],'Submission run incomplete')
    require(launch['schema']=='digit-ig-window-launch-v1' and launch['mode']==state['mode'],'Wrong launch')
    require(launch['candidate_sha256']==state['candidate_sha256']==sha(HERE/'manifest.json'),'Wrong submission version')
    require(launch['worker_candidate_sha256']==state['worker_candidate_sha256']==binding['candidate_sha256']==sha(ROOT/'candidates/ig_perf_v5/manifest.json'),'Wrong worker version')
    from submission.v9.common import verify as verify_entry
    require(launch['entry_manifest_sha256']==state['entry_manifest_sha256']==verify_entry(),'Wrong submission entry')
    require(launch['input_binding_sha256']==sha(folder/'inputs.json') and launch['protocol_sha256']==sha(P),'Wrong input/protocol binding')
    require(launch['gpu']==state['gpu'] and launch['monitor_policy']==POLICY,'Wrong GPU/monitor policy')
    require(launch['dataset']==state['dataset']=='IG' and launch['model']==state['model'] and state['model'] in cfg()['models'] and launch['seed']==state['seed']==0,'Wrong dataset/model/seed')
    require(launch['raw_ssd_writes'] is False and state['raw_ssd_writes'] is False,'Unexpected raw write')
    jobs=plan(state['mode'],state['model']);require(launch['plan']==jobs,'Unexpected job plan')
    require([(w['mode'],w['arm']) for w in state['workers']]==[(j['mode'],j['arm']) for j in jobs],'Missing/duplicate workers')
    check_inputs(binding);phases={}
    for job,w in zip(jobs,state['workers']):
        expected=worker_command(job,folder)
        require(w['command'][1:]==expected[1:] and w['model']==state['model'],'Worker command/model differs')
        prefix=job['mode']+'_'+job['arm'];receipt=read(folder/(prefix+'_receipt.json'));accepted=folder/(prefix+'_accepted.json')
        require(receipt['passed'] and receipt['job']==job and receipt['accepted_sha256']==sha(accepted),'Worker acceptance receipt differs')
        require(receipt['report_sha256']==sha(folder/job['mode']/job['arm']/'report.json'),'Worker report differs')
        from candidates.ig_perf_v5.controller import arm_evidence
        require(receipt['files']==arm_evidence(folder,job),'Worker/monitor evidence changed or missing')
        r=review_worker(folder,w,state['pid'],binding,state['gpu']);require(read(accepted)==r and r['model_name']==state['model'],'Accepted report/model differs')
        phases.setdefault(job['mode'],{})[job['arm']]=r
    for phase,reports in phases.items():
        saved=read(folder/(phase+'_summary.json'));pair=pair_check(reports,phase=='smoke')
        require(all(saved[k]==v for k,v in pair.items()),'Phase summary differs')
        strict=all(r['external_monitor']['strict_monitor_passed'] for r in reports.values())
        require(saved['strict_resource_acceptance']==strict,'Phase monitor status differs')
        for arm in reports:require(saved['report_sha256'][arm]==sha(folder/(phase+'_'+arm+'_accepted.json')),'Phase report hash differs')
    if state['mode']=='smoke':
        require(allow_smoke,'Smoke has no timed-window performance metrics')
        require(state['strict_resource_acceptance'] and not state['training_performance_io_passed'],'Wrong smoke scope')
        return dict(passed=True,scope='native smoke only',new_full_training=False,test_calls=0)
    value=comparison(phases['benchmark'],'submission_v9_nvml_ig_fresh_native_pair',dict(directory=str(folder),launch_sha256=sha(folder/'launch.json')))
    require(value['strict_resource_acceptance']==state['strict_resource_acceptance'] and state['training_performance_io_passed'],'Final acceptance differs')
    return value


def emit(value,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write(output/'summary.json',value)
    keys=['dataset','model','arm','updates','warmup_batches','measured_batches','measurement_windows','training_seconds','seconds_per_batch','training_roots_per_second','warmup_seconds','setup_seconds','worker_seconds','window_relative_spread','outer_layer_group_edge_share','cpu_feature_request_share','observed_peak_gpu_bytes']
    rows=[];window_rows=[]
    for r in value['rows']:
        for phase,io in r['io'].items():
            row={k:r[k] for k in keys};row.update(phase=phase,monitor_timeout_count=r['monitor']['timeout_count']);row.update(io);rows.append(row)
        for w in r['windows']:window_rows.append(dict(dataset=r['dataset'],model=r['model'],arm=r['arm'],**{k:v for k,v in w.items() if k!='io'},**w['io']))
    for filename,records in [('summary.csv',rows),('windows.csv',window_rows)]:
        fields=list(dict.fromkeys(k for row in records for k in row))
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    a,b=value['rows'];q=value['ratios'];lines=['# IG / '+a['model'].upper()+' bounded training windows','', 'Per arm: 20 warmup batches + 3 x 30 timed batches. No validation, test, accuracy or full-epoch claim. Strict resource acceptance: '+str(value['strict_resource_acceptance'])+'.','', '| Metric | GIDS | DiGiT |','|---|---:|---:|']
    def number(x):return 'NA' if x is None else '%.4f'%x
    for label,key in [('90 measured batches (s)','training_seconds'),('Mean seconds / batch','seconds_per_batch'),('Training roots / second','training_roots_per_second'),('20 warmup batches (s)','warmup_seconds'),('Setup (s)','setup_seconds'),('Worker including setup (s)','worker_seconds'),('Window relative spread','window_relative_spread'),('Group edge share','outer_layer_group_edge_share')]:lines.append('| %s | %s | %s |'%(label,number(a[key]),number(b[key])))
    for j in range(3):lines.append('| Window %d / 30 batches (s) | %s | %s |'%(j+1,number(a['windows'][j]['seconds']),number(b['windows'][j]['seconds'])))
    for label,key in [('Measured physical SSD bytes','ssd_completed_bytes'),('Measured useful SSD GB/s','ssd_useful_gbps'),('Measured physical SSD GB/s','ssd_physical_gbps'),('Logical feature supply GB/s','effective_feature_gbps')]:lines.append('| %s | %s | %s |'%(label,number(a['io']['training'][key]),number(b['io']['training'][key])))
    lines+=['','Bounded training speedup: %.4fx. All three consecutive windows are included.'%q['training_speedup'],'',*['- '+s for s in value['limitations']]]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',choices=['sage','gcn','gat'],required=True);a=p.parse_args();output=output_path(a.output);verify();value=load_formal(a.input);require(all(r['model']==a.model for r in value['rows']),'Requested/result model differs');emit(value,output);print(json.dumps(dict(passed=True,output=str(output)),indent=2))
if __name__=='__main__':main()
