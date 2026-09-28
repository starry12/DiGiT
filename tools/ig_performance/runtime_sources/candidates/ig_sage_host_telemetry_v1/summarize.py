"""Rebuild actual worker evidence and retain both independent performance runs."""
import csv
import statistics
import numpy as np
from .common import *
from .validation import pair_check
from candidates.ig_perf_window300_v1.summarize import metrics


def same_arm(a,b,af,bf):
    import torch
    for key in ('arm','model_config','optimizer','initial_parameters_sha256','seed','updates','measured_batches'):
        require(a[key]==b[key],'Repeat protocol differs: '+key)
    for phase in ('warmup','training'):
        require(a[phase]['roots_sha256']==b[phase]['roots_sha256'],'Repeat roots differ')
    left=torch.load(Path(af)/'final_model.pt',map_location='cpu')
    right=torch.load(Path(bf)/'final_model.pt',map_location='cpu')
    require(set(left['model'])==set(right['model']),'Repeat parameter keys differ')
    aa=np.asarray(a['warmup']['losses']+a['training']['losses'])
    bb=np.asarray(b['warmup']['losses']+b['training']['losses'])
    error=max(float((v-right['model'][k]).abs().max()) for k,v in left['model'].items())
    return dict(losses_exact=bool(np.array_equal(aa,bb)),
        max_abs_loss_difference=float(np.max(np.abs(aa-bb))),
        final_parameters_exact=a['final_parameters_sha256']==b['final_parameters_sha256'],
        max_abs_parameter_difference=error,
        adam_exact=state_hash(left['optimizer'])==state_hash(right['optimizer']),
        sampled_shape_totals_equal=all(a[p]['sampling_shape_totals']==b[p]['sampling_shape_totals'] for p in ('warmup','training')),
        note='Same-system numerical observations; retain any GPU nondeterminism. Cross-system losses are not required equal.')


def comparisons(reports,folder):
    from .smoke_evidence import review as review_smokes
    smoke=review_smokes()
    require(smoke==read(Path(folder)/'inherited_smokes.json')==read(OUT/'inherited_smokes.json'),'Inherited smoke evidence changed')
    pairs={str(i):pair_check(dict(gids=reports['gids_'+str(i)],digit_full=reports['digit_'+str(i)]),False) for i in (1,2)}
    repeats={}
    for arm,prefix in [('gids','gids'),('digit_full','digit')]:
        a,b=prefix+'_1',prefix+'_2'
        repeats[arm]=same_arm(reports[a],reports[b],Path(folder)/a/arm,Path(folder)/b/arm)
    return dict(smoke=smoke,pairs=pairs,same_arm_repeats=repeats)


def verify_version(launch,state,binding,actual):
    require(launch['candidate_sha256']==state['candidate_sha256']==binding['candidate_sha256']==actual,'Wrong actual worker version')


def load_formal(folder):
    from .controller import plan,worker_command,arm_evidence
    from .review import review_worker
    folder=Path(folder);state=read(folder/'status.json');launch=read(folder/'launch.json');binding=read(folder/'inputs.json')
    require(state['schema']=='digit-ig-host-telemetry-run-v1' and state['passed'] and state['complete'],'Run incomplete')
    require(launch['schema']=='digit-ig-host-telemetry-launch-v1' and launch['mode']==state['mode'],'Wrong launch')
    verify_version(launch,state,binding,verify())
    require(launch['input_binding_sha256']==sha(folder/'inputs.json') and launch['protocol_sha256']==sha(P) and launch['optimization_protocol_sha256']==sha(HERE/'optimization_protocol.json'),'Protocol/input mismatch')
    require(state['model']==launch['model']=='sage' and state['dataset']==launch['dataset']=='IG' and state['gpu']==launch['gpu']==2 and state['seed']==launch['seed']==0,'Scope mismatch')
    require(launch['monitor_policy']==POLICY and not launch['raw_ssd_writes'] and not state['raw_ssd_writes'],'Runtime policy mismatch')
    jobs=plan(state['mode'],state['model']);require(launch['plan']==jobs and len(state['workers'])==len(jobs),'Wrong worker plan')
    check_inputs(binding);reports={};counts={}
    for j,w in zip(jobs,state['workers']):
        require(all(w[k]==v for k,v in j.items()),'Unexpected worker')
        require(w['command'][1:]==worker_command(j,folder)[1:],'Wrong command')
        prefix=j['mode']+'_'+j['arm'];receipt=read(folder/(prefix+'_receipt.json'));accepted=folder/(prefix+'_accepted.json')
        require(receipt['passed'] and receipt['job']==j and receipt['files']==arm_evidence(folder,j),'Changed/incomplete evidence')
        require(receipt['accepted_sha256']==sha(accepted) and receipt['report_sha256']==sha(folder/j['mode']/j['arm']/'report.json'),'Changed report')
        r=review_worker(folder,w,state['pid'],binding,state['gpu']);require(r==read(accepted),'Rebuilt acceptance differs')
        reports[j['mode']]=r;counts[prefix]=len(receipt['files'])
    compared=comparisons(reports,folder)
    require(compared==read(folder/'comparisons.json'),'Rebuilt pair/repeat comparisons differ')
    rows=[]
    for j in jobs:
        if j['smoke']:continue
        r=reports[j['mode']];row=metrics(j['arm'],r)
        row.update(run=j['mode'],profile_mode=r['stage_profile']['mode'],cpu_policy=r['affinity']['initial']['policy'])
        spans={}
        for w in r['stage_profile']['windows']:
            if w['phase']=='training':
                for key,v in w['spans'].items():spans[key]=spans.get(key,0.)+v['inclusive_seconds']
        row.update(host_spans_seconds=spans,training_telemetry=r['training_telemetry'])
        rows.append(row)
    means={}
    for arm in ('gids','digit_full'):
        rr=[r for r in rows if r['arm']==arm]
        require(len(rr)==2,'Each arm needs two independent runs')
        means[arm]=dict(training_seconds=statistics.mean(r['training_seconds'] for r in rr),
            setup_seconds=statistics.mean(r['setup_seconds'] for r in rr),
            warmup_seconds=statistics.mean(r['warmup_seconds'] for r in rr),
            worker_seconds=statistics.mean(r['worker_seconds'] for r in rr),
            ssd_completed_bytes=statistics.mean(r['io']['training']['ssd_completed_bytes'] for r in rr),
            runs_seconds=[r['training_seconds'] for r in rr])
    gids,digit=means['gids'],means['digit_full']
    strict=all(r['external_monitor']['strict_monitor_passed'] for r in reports.values())
    require(strict==state['strict_resource_acceptance'] and state['training_performance_io_passed'],'Final monitor acceptance differs')
    diagnostics=all(r['training_telemetry']['diagnostic_complete'] for r in reports.values())
    require(diagnostics==state['diagnostic_complete'],'Final diagnostic coverage differs')
    return dict(schema='digit-ig-host-telemetry-results-v1',passed=True,complete=True,source=str(folder),candidate_sha256=verify(),
        arms=['gids','digit_full'],rows=rows,means=means,comparisons=compared,
        ratios=dict(mean_training_speedup=gids['training_seconds']/digit['training_seconds'],
            mean_training_time_reduction_percent=100*(1-digit['training_seconds']/gids['training_seconds']),
            mean_worker_speedup=gids['worker_seconds']/digit['worker_seconds'],
            mean_ssd_bytes_reduction_percent=100*(1-digit['ssd_completed_bytes']/gids['ssd_completed_bytes'])),
        strict_resource_acceptance=strict,evidence_files_by_worker=counts,independent_runs_per_arm=2,
        diagnostic_complete=diagnostics,
        epoch_time_claim=False,accuracy_claim=False,
        limits=['GIDS uses unchanged default scheduling; original DiGiT threads use logical CPU2. Speedup includes this scheduling difference.',
            'Both use exactly the previous host stage instrumentation. Original sampling/model/cache/native I/O retained; no EID/frontier optimization.',
            'External CPU/GPU scheduling and clock observations are aligned to original timing windows; missing coverage is reported separately from training acceptance.',
            'Fresh-process ABBA order, one seed, two runs per system. Retain all 100-batch windows; windows within a run are not independent repeats.',
            'Timed training uses 20 warmup plus 300 measured batches. Setup and warmup are separate, with no epoch, validation, test or accuracy claim.',
            'No standalone kernel-time claim or stable/multi-seed speedup claim. PA and AE remain unchanged.'])


def emit(value,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write(output/'summary.json',value)
    rows=[dict(run=r['run'],arm=r['arm'],cpu_policy=r['cpu_policy'],training_seconds=r['training_seconds'],
        setup_seconds=r['setup_seconds'],warmup_seconds=r['warmup_seconds'],worker_seconds=r['worker_seconds'],
        ssd_completed_bytes=r['io']['training']['ssd_completed_bytes'],
        window_seconds=[w['seconds'] for w in r['windows']]) for r in value['rows']]
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# IG/SAGE original DiGiT CPU2 versus GIDS','',
        '20 warmup + 300 timed batches per process. GIDS default scheduling; DiGiT CPU2.',
        '', '| Run | CPU policy | Training (s) | Setup (s) | Warmup (s) | SSD read (GB) |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| %s | %s | %.2f | %.2f | %.2f | %.2f |'%(r['run'],r['cpu_policy'],r['training_seconds'],r['setup_seconds'],r['warmup_seconds'],r['ssd_completed_bytes']/1e9))
    lines+=['','GIDS/DiGiT mean training: %.2f / %.2f s; speedup %.3fx; time reduction %.2f%%.'%(
        value['means']['gids']['training_seconds'],value['means']['digit_full']['training_seconds'],
        value['ratios']['mean_training_speedup'],value['ratios']['mean_training_time_reduction_percent']),'']
    lines.extend('- %s: 100-batch windows %s s.'%(r['run'],', '.join('%.2f'%x for x in r['window_seconds'])) for r in rows)
    lines+=['',*['- '+v for v in value['limits']]]
    lines+=['','Diagnostic telemetry complete: '+str(value['diagnostic_complete']),'',
        '| Run | Sampling (s) | Feature fetch (s) | Forward (s) | Backward (s) | Adam (s) |',
        '|---|---:|---:|---:|---:|---:|']
    stage_rows=[]
    for r in value['rows']:
        s=r['host_spans_seconds']
        lines.append('| %s | %.2f | %.2f | %.2f | %.2f | %.2f |'%(r['run'],s['sampling'],s['feature_fetch'],s['forward'],s['backward'],s['adam']))
        stage_rows.extend(dict(run=r['run'],stage=k,seconds=v,nested='/' in k) for k,v in s.items())
    with (output/'stages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run','stage','seconds','nested']);writer.writeheader();writer.writerows(stage_rows)
    write(output/'telemetry.json',{r['run']:r['training_telemetry'] for r in value['rows']})
    (output/'README.md').write_text('\n'.join(lines)+'\n')
