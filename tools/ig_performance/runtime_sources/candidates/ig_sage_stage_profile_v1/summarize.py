"""Rebuild every receipt and report symmetric stage times and off controls."""
import csv
import math
import numpy as np
from .common import *
from .validation import pair_check
from candidates.ig_perf_window300_v1.summarize import metrics

def load_formal(folder,allow_smoke=False):
    from candidates.ig_sage_stage_profile_v1.controller import plan,worker_command
    from candidates.ig_sage_stage_profile_v1.review import review_worker
    folder=Path(folder).resolve();state=read(folder/'status.json');launch=read(folder/'launch.json');binding=read(folder/'inputs.json')
    require(state['schema']=='digit-ig-window-run-v1' and state['passed'] and state['complete'],'Submission run incomplete')
    require(launch['schema']=='digit-ig-window-launch-v1' and launch['mode']==state['mode'],'Wrong launch')
    require(launch['candidate_sha256']==state['candidate_sha256']==sha(HERE/'manifest.json'),'Wrong submission version')
    require(launch['worker_candidate_sha256']==state['worker_candidate_sha256']==binding['candidate_sha256']==sha(ROOT/'candidates/ig_perf_window300_v1/manifest.json'),'Wrong worker version')
    from candidates.ig_sage_stage_profile_v1.common import verify as verify_entry
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
        from candidates.ig_sage_stage_profile_v1.controller import arm_evidence
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
    value=comparison(phases,folder)
    require(value['strict_resource_acceptance']==state['strict_resource_acceptance'] and state['training_performance_io_passed'],'Final acceptance differs')
    return value


def compare_same_arm(off, host, folder, arm):
    import torch
    for key in ('model_config', 'optimizer', 'initial_parameters_sha256', 'updates', 'seed'):
        require(off[key] == host[key], 'Off/host protocol differs: '+key)
    for phase in ('warmup', 'training'):
        require(off[phase]['roots_sha256'] == host[phase]['roots_sha256'], 'Off/host roots differ')
    a = np.asarray(off['warmup']['losses']+off['training']['losses'])
    b = np.asarray(host['warmup']['losses']+host['training']['losses'])
    left = torch.load(folder/'control'/arm/'final_model.pt', map_location='cpu')
    right = torch.load(folder/'profile'/arm/'final_model.pt', map_location='cpu')
    require(set(left['model']) == set(right['model']), 'Off/host parameter keys differ')
    error = max(float((v-right['model'][key]).abs().max()) for key,v in left['model'].items())
    adam_equal = state_hash(left['optimizer']) == state_hash(right['optimizer'])
    return dict(initial_parameters_equal=True, frozen_roots_equal=True,
        all_losses_exact=bool(np.array_equal(a,b)), max_abs_loss_difference=float(np.max(np.abs(a-b))),
        losses_close_rtol1e_5_atol1e_6=bool(np.allclose(a,b,rtol=1e-5,atol=1e-6)),
        final_parameters_exact=off['final_parameters_sha256']==host['final_parameters_sha256'],
        max_abs_parameter_difference=error, adam_state_exact=adam_equal,
        sampled_shape_totals_equal=all(off[p]['sampling_shape_totals']==host[p]['sampling_shape_totals'] for p in ('warmup','training')),
        note='Numerical comparisons are observations, not assertions of GPU determinism. Any mismatch is retained.')


def comparison(phases, folder):
    from .profile import TOP
    require(set(phases)=={'smoke','control','profile'}, 'Incomplete symmetric diagnostic')
    rows=[]; overhead={}; parity={}; stages={}; window_rows=[]
    for mode in ('control','profile'):
        pair_check(phases[mode],False)
        for arm in ('gids','digit_full'):
            r=phases[mode][arm]
            row=metrics(arm,r);row.update(run_phase=mode,profile_mode=r['stage_profile']['mode'])
            rows.append(row)
            for w in r['stage_profile']['windows']:
                window_rows.append(dict(arm=arm,profile_mode=r['stage_profile']['mode'],**w))
    for arm in ('gids','digit_full'):
        off,host=[phases[m][arm] for m in ('control','profile')]
        overhead[arm]=dict(off_seconds=off['training_seconds'],host_seconds=host['training_seconds'],
            difference_seconds=host['training_seconds']-off['training_seconds'],
            difference_percent=100*(host['training_seconds']/off['training_seconds']-1),
            interpretation='Includes instrumentation and fresh-process machine/order variation; do not subtract as an exact correction.')
        parity[arm]=compare_same_arm(off,host,folder,arm)
        totals={}
        for window in host['stage_profile']['windows']:
            if window['phase']!='training':continue
            for key,value in window['spans'].items():
                target=totals.setdefault(key,dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
                for k in target:target[k]+=value[k]
        stages[arm]=totals
    stage_rows=[]
    for key in list(TOP)+sorted((set(stages['gids'])|set(stages['digit_full']))-set(TOP)):
        if any(row['stage']==key for row in stage_rows):continue
        a=stages['gids'].get(key,{});b=stages['digit_full'].get(key,{})
        stage_rows.append(dict(stage=key,nested='/' in key,
            gids_seconds=a.get('inclusive_seconds',0.),digit_seconds=b.get('inclusive_seconds',0.),
            digit_minus_gids_seconds=b.get('inclusive_seconds',0.)-a.get('inclusive_seconds',0.),
            gids_exclusive_seconds=a.get('exclusive_seconds',0.),digit_exclusive_seconds=b.get('exclusive_seconds',0.),
            gids_calls=a.get('calls',0),digit_calls=b.get('calls',0)))
    strict=all(r['external_monitor']['strict_monitor_passed'] for phase in phases.values() for r in phase.values())
    return dict(schema='digit-ig-symmetric-stage-results-v1',passed=True,complete=True,
        source=dict(directory=str(folder),launch_sha256=sha(folder/'launch.json')),
        candidate_sha256=verify(),protocol_sha256=sha(P),profile_protocol_sha256=sha(HERE/'profile_protocol.json'),
        strict_resource_acceptance=strict,rows=rows,stages=stage_rows,windows=window_rows,
        instrumentation_overhead=overhead,same_arm_comparisons=parity,
        loss_or_parameter_difference_observed=any(not (v['all_losses_exact'] and v['final_parameters_exact'] and v['adam_state_exact']) for v in parity.values()),
        added_cuda_synchronization=False,epoch_time_claim=False,final_accuracy_claim=False,
        limitations=['Host wall time includes asynchronous dispatch and waits at unchanged original sites; it is not standalone CUDA kernel time.',
            'Nested stages are already included in parents. Only top-level stages plus the uninstrumented residual reconcile to each synchronized E2E window.',
            'One independent process per system/mode. Three consecutive windows are not independent repeats. Off/host differences include machine and run-order variability.',
            'All original graph/model/native/cache/monitor/CPU-affinity settings are retained. No PA optimization or AE change.',
            'Setup and 20 warmup batches are separate from the 300 measured batches. No full epoch, validation, test or accuracy claim.'])


def emit(value, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'summary.json',value)
    with (output/'stages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(value['stages'][0]));writer.writeheader();writer.writerows(value['stages'])
    lines=['# IG/SAGE symmetric host-stage diagnostic','',
        '20 warmup + 300 timed batches per formal worker. No validation, test, accuracy or full-epoch claim.',
        '', '| Mode | GIDS (s) | DiGiT (s) |','|---|---:|---:|']
    for label,key in [('Off control','off_seconds'),('Host profile','host_seconds'),('Host minus off','difference_seconds')]:
        lines.append('| %s | %.2f | %.2f |'%(label,value['instrumentation_overhead']['gids'][key],value['instrumentation_overhead']['digit_full'][key]))
    lines+=['','| Top-level host stage | GIDS (s) | DiGiT (s) | DiGiT minus GIDS (s) |','|---|---:|---:|---:|']
    for row in value['stages']:
        if not row['nested']:lines.append('| {stage} | {gids_seconds:.2f} | {digit_seconds:.2f} | {digit_minus_gids_seconds:.2f} |'.format(**row))
    residual={a:sum(w['uninstrumented_seconds'] for w in value['windows'] if w['arm']==a and w['profile_mode']=='host' and w['phase']=='training') for a in ('gids','digit_full')}
    lines.append('| Uninstrumented residual | %.2f | %.2f | %.2f |'%(residual['gids'],residual['digit_full'],residual['digit_full']-residual['gids']))
    lines+=['','Nested detail (already included above):','', '| Stage | GIDS (s) | DiGiT (s) |','|---|---:|---:|']
    for row in value['stages']:
        if row['nested']:lines.append('| {stage} | {gids_seconds:.2f} | {digit_seconds:.2f} |'.format(**row))
    lines+=['','Same-arm numerical comparisons:','']
    for arm,v in value['same_arm_comparisons'].items():
        lines.append('- %s: exact losses=%s, exact final parameters=%s, exact Adam=%s; max absolute loss/parameter difference=%.8g/%.8g; sampled shapes equal=%s.'%(arm,v['all_losses_exact'],v['final_parameters_exact'],v['adam_state_exact'],v['max_abs_loss_difference'],v['max_abs_parameter_difference'],v['sampled_shape_totals_equal']))
    lines+=['',*['- '+s for s in value['limitations']]]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
