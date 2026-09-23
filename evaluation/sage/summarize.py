"""Revalidate accepted historical evidence or new v2 runs, then emit honest phase metrics."""
import argparse,csv,json,math,sys
from pathlib import Path
from evaluation.sage.common import *
from runtime.io.accounting import summarize as io_summary
from training.sage.validation import pair_check
from training.sage.worker import check_inputs

def phase_metrics(regions,phase):
    r=io_summary(regions);seconds=r.pop('train_seconds');per=r.pop('ssd_useful_per_train_gbps')
    r.update(phase=phase,wall_seconds=seconds,ssd_useful_per_wall_gbps=per)
    r['logical_feature_per_wall_gbps']=r['logical_feature_bytes']/seconds/1e9
    return r

def metrics(arm,r):
    train=sum(e['train_seconds'] for e in r['epochs']);valid=sum(e['validation']['seconds'] for e in r['epochs'])
    require(math.isclose(train,r['training_seconds'],rel_tol=1e-12) and math.isclose(valid,r['validation_seconds'],rel_tol=1e-12),'Timing totals differ')
    test=r['test'];require(test is not None,'Full summary cannot use a smoke report')
    phases=dict(training=phase_metrics([(e['training'],e['train_seconds']) for e in r['epochs']],'training'),
                validation=phase_metrics([(e['validation'],e['validation']['seconds']) for e in r['epochs']],'validation'),
                test=phase_metrics([(test,test['seconds'])],'test'))
    windows=[w for e in r['epochs'] for w in e['windows']];cpu=sum(e['training']['feature']['cpu'] for e in r['epochs']);gpu=sum(e['training']['feature']['gpu_ssd'] for e in r['epochs'])
    group=sum(w['group_edges'] for w in windows);edges=sum(w['actual_edges'] for w in windows)
    return dict(dataset='PA',model='sage',seed=0,repeat=0,arm=arm,epochs=len(r['epochs']),updates=r['updates'],
                training_seconds=train,mean_training_epoch_seconds=train/len(r['epochs']),order_excluded_training_seconds=sum(e['train_seconds']-e['order_seconds'] for e in r['epochs']),
                validation_seconds=valid,test_seconds=test['seconds'],online_seconds=train+valid+test['seconds'],
                setup_seconds=r['setup_seconds'],metadata_setup_seconds=r['metadata_setup_seconds'],worker_seconds=r['worker_seconds'],
                validation_accuracy_epoch20=r['epochs'][-1]['validation']['accuracy'],test_accuracy=test['accuracy'],test_examples=test['examples'],
                cpu_feature_request_share=cpu/(cpu+gpu),outer_layer_group_edge_share=group/edges,outer_layer_group_edges=group,outer_layer_sampled_edges=edges,
                io=phases,observed_peak_gpu_bytes=r['external_monitor']['observed_peak_device_used_bytes'],monitor=r['external_monitor'],
                worker_candidate_sha256=r['candidate_sha256'],final_parameters_sha256=r['final_parameters_sha256'])

def comparison(reports,provenance,source):
    pair_check(reports,False);rows=[metrics(arm,reports[arm]) for arm in ('gids','digit_full')];a,b=rows
    strict=all(r['monitor']['strict_monitor_passed'] for r in rows)
    return dict(schema='digit-submission-results-v2',passed=True,complete=True,training_accuracy_io_passed=True,strict_resource_acceptance=strict,
        qualification='complete' if strict else 'complete_with_monitoring_gaps',provenance=provenance,source=source,rows=rows,
        ratios=dict(training_speedup=a['training_seconds']/b['training_seconds'],online_speedup=a['online_seconds']/b['online_seconds'],worker_speedup=a['worker_seconds']/b['worker_seconds'],
                    test_accuracy_difference_percentage_points=100*(b['test_accuracy']-a['test_accuracy']),
                    training_ssd_bytes_reduction_percent=100*(1-b['io']['training']['ssd_completed_bytes']/a['io']['training']['ssd_completed_bytes'])),
        seed_count=1,repeat_count=1,full_submission_matrix_complete=False,paper_accuracy_reproduced=False,
        limitations=['Only reconstructed bidirectional PA/SAGE, seed0; inherited g2 layout and no BFS.',
                     'Original paper accuracy values are not substituted for measured results; no statistical accuracy claim.',
                     'Sampled GPU peaks are lower bounds; any timeout gaps and strict-resource status are retained.'],
        units=dict(seconds='seconds',accuracy='fraction [0,1]',throughput='decimal GB/s',bytes='bytes'),
        timing_note='Online=train+all validation+final test; worker also includes setup/checks, excludes controller input hashing and teardown. SSD GB/s uses SSD-active time, logical feature GB/s uses feature-call time.')

def load_accepted():
    raise RuntimeError('Use the root reference command for archived PA results; new runs require --input')

def load_formal(folder,allow_smoke=False):
    from evaluation.sage.controller import plan,worker_command
    from evaluation.sage.review import review_worker
    folder=Path(folder).resolve();state=read(folder/'status.json');launch=read(folder/'launch.json');binding=read(folder/'inputs.json')
    require(state['schema']=='digit-submission-run-v2' and state['passed'] and state['complete'],'Submission run incomplete')
    require(launch['schema']=='digit-submission-launch-v2' and launch['mode']==state['mode'],'Wrong launch')
    require(launch['submission_sha256']==state['submission_sha256']==sha(HERE/'manifest.json'),'Wrong submission version')
    require(launch['worker_candidate_sha256']==state['worker_candidate_sha256']==binding['candidate_sha256']==sha(ROOT/'training/sage/manifest.json'),'Wrong worker version')
    require(launch['input_binding_sha256']==sha(folder/'inputs.json') and launch['protocol_sha256']==sha(HERE/'pa_sage_protocol.json'),'Wrong input/protocol binding')
    require(launch['gpu']==state['gpu'] and launch['monitor_policy']==POLICY,'Wrong GPU/monitor policy')
    require(launch['dataset']==state['dataset']=='PA' and launch['model']==state['model']=='sage' and launch['seed']==state['seed']==0,'Wrong dataset/model/seed')
    require(launch['raw_ssd_writes'] is False and state['raw_ssd_writes'] is False,'Unexpected raw write')
    jobs=plan(state['mode']);require(launch['plan']==jobs,'Unexpected job plan')
    require([(w['mode'],w['arm']) for w in state['workers']]==[(j['mode'],j['arm']) for j in jobs],'Missing/duplicate workers')
    check_inputs(binding);phases={}
    for job,w in zip(jobs,state['workers']):
        expected=worker_command(job,folder)
        require(w['command'][1:]==expected[1:],'Worker command differs')
        prefix=job['mode']+'_'+job['arm'];receipt=read(folder/(prefix+'_receipt.json'));accepted=folder/(prefix+'_accepted.json')
        require(receipt['passed'] and receipt['job']==job and receipt['accepted_sha256']==sha(accepted),'Worker acceptance receipt differs')
        require(receipt['report_sha256']==sha(folder/job['mode']/job['arm']/'report.json'),'Worker report differs')
        from evaluation.sage.controller import arm_evidence
        require(receipt['files']==arm_evidence(folder,job),'Worker/monitor evidence changed or missing')
        r=review_worker(folder,w,state['pid'],binding,state['gpu']);require(read(accepted)==r,'Accepted report differs')
        phases.setdefault(job['mode'],{})[job['arm']]=r
    for phase,reports in phases.items():
        saved=read(folder/(phase+'_summary.json'));pair=pair_check(reports,phase=='smoke')
        require(all(saved[k]==v for k,v in pair.items()),'Phase summary differs')
        strict=all(r['external_monitor']['strict_monitor_passed'] for r in reports.values())
        require(saved['strict_resource_acceptance']==strict,'Phase monitor status differs')
        for arm in reports:require(saved['report_sha256'][arm]==sha(folder/(phase+'_'+arm+'_accepted.json')),'Phase report hash differs')
    if state['mode']=='smoke':
        require(allow_smoke,'Smoke has no complete training/test metrics')
        require(state['strict_resource_acceptance'] and not state['training_accuracy_io_passed'],'Wrong smoke scope')
        return dict(passed=True,scope='native smoke only',new_full_training=False,test_calls=0)
    value=comparison(phases['full'],'submission_v2_fresh_native_pair',dict(directory=str(folder),launch_sha256=sha(folder/'launch.json')))
    require(value['strict_resource_acceptance']==state['strict_resource_acceptance'] and state['training_accuracy_io_passed'],'Final acceptance differs')
    return value

def emit(value,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'summary.json',value)
    base=['dataset','model','seed','repeat','arm','epochs','updates','training_seconds','mean_training_epoch_seconds','validation_seconds','test_seconds','online_seconds','setup_seconds','metadata_setup_seconds','worker_seconds','test_accuracy','validation_accuracy_epoch20','observed_peak_gpu_bytes','worker_candidate_sha256']
    fields=['logical_feature_bytes','gpu_feature_bytes','ssd_useful_bytes','ssd_completed_bytes','ssd_primary_bytes','ssd_replay_bytes','ssd_active_seconds','feature_seconds','wall_seconds','ssd_useful_gbps','ssd_physical_gbps','effective_feature_gbps','ssd_payload_utilization','ssd_useful_per_wall_gbps','logical_feature_per_wall_gbps']
    csv_rows=[]
    for r in value['rows']:
        for phase,v in r['io'].items():
            row={k:r[k] for k in base};row.update({k:v[k] for k in fields});row.update(phase=phase,strict_resource_acceptance=r['monitor']['strict_monitor_passed'],monitor_timeout_count=r['monitor']['timeout_count']);csv_rows.append(row)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=base+['phase']+fields+['strict_resource_acceptance','monitor_timeout_count']);w.writeheader();w.writerows(csv_rows)
    a,b=value['rows'];q=value['ratios']
    def number(value,scale=1):return 'NA' if value is None else '%.4f'%(value*scale)
    lines=['# PA / GraphSAGE measured results','', 'Provenance: `'+value['provenance']+'`.', '',
           'Both systems completed seed0 / 20 epochs / one final test. Strict resource acceptance: **'+str(value['strict_resource_acceptance']).lower()+'**. Single-seed results do not establish statistical accuracy improvement.','',
           '| Metric | GIDS | DiGiT |','|---|---:|---:|']
    specs=[('Training (s)','training_seconds',1),('Mean epoch (s)','mean_training_epoch_seconds',1),('Validation total (s)','validation_seconds',1),('Final test (s)','test_seconds',1),('Train + valid + test (s)','online_seconds',1),('Worker including setup/checks (s)','worker_seconds',1),('Test accuracy (%)','test_accuracy',100)]
    for title,key,scale in specs:lines.append('| %s | %s | %s |'%(title,number(a[key],scale),number(b[key],scale)))
    for title,key,scale in [('Training physical SSD (TB)','ssd_completed_bytes',1e-12),('Training useful SSD (GB/s)','ssd_useful_gbps',1),('Training physical SSD (GB/s)','ssd_physical_gbps',1),('Logical feature supply (GB/s)','effective_feature_gbps',1)]:
        lines.append('| %s | %s | %s |'%(title,number(a['io']['training'][key],scale),number(b['io']['training'][key],scale)))
    lines+=['', 'Training speedup: %.4fx; online speedup: %.4fx; worker speedup: %.4fx.'%(q['training_speedup'],q['online_speedup'],q['worker_speedup']), '',
            'SSD GB/s uses summed active time. Feature GB/s includes CPU/GPU hits and uses feature-call time. CSV contains separate training, validation and test rows; JSON retains all raw numerators/denominators.', '',
            'GPU monitoring timeouts: GIDS %d, DiGiT %d. Sampled peaks are lower bounds.'%(a['monitor']['timeout_count'],b['monitor']['timeout_count']), '', *['- '+s for s in value['limitations']]]
    (output/'README.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--accepted',action='store_true');g.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=output_path(a.output);verify();value=load_accepted() if a.accepted else load_formal(a.input);emit(value,output)
    print(json.dumps(dict(passed=True,provenance=value['provenance'],strict_resource_acceptance=value['strict_resource_acceptance'],output=str(output)),indent=2))
if __name__=='__main__':main()
