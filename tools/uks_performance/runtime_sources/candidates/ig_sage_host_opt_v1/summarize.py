"""Independent variant acceptance and deterministic export; no parent-worker alias."""
import csv
import numpy as np
from .common import *


def compare_reports(a,b,af,bf):
    import torch
    for key in ('arm','model_config','optimizer','initial_parameters_sha256','seed','updates','measured_batches'):
        require(a[key]==b[key],'Variant protocol differs: '+key)
    for phase in ('warmup','training'):
        for key in ('roots_sha256','sampling_shape_totals','feature_rows','group_edges','outer_edges'):
            require(a[phase][key]==b[phase][key],'Variant sampled extent differs: '+phase+'/'+key)
    aa=np.asarray(a['warmup']['losses']+a['training']['losses']);bb=np.asarray(b['warmup']['losses']+b['training']['losses'])
    require(np.allclose(aa,bb,rtol=1e-5,atol=1e-6),'Loss sequence differs')
    left=torch.load(Path(af)/'final_model.pt',map_location='cpu');right=torch.load(Path(bf)/'final_model.pt',map_location='cpu')
    def compare(x,y):
        if torch.is_tensor(x):
            require(torch.is_tensor(y) and x.shape==y.shape and x.dtype==y.dtype,'State tensor type/shape differs')
            require(torch.allclose(x,y,rtol=1e-5,atol=1e-6),'State tensor values differ')
        elif isinstance(x,dict):
            require(set(x)==set(y),'State keys differ')
            for key in x:compare(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            require(len(x)==len(y),'State sequence differs')
            for xx,yy in zip(x,y):compare(xx,yy)
        else:require(x==y,'State scalar differs')
    compare(left['model'],right['model']);compare(left['optimizer'],right['optimizer'])
    return dict(passed=True,reference=a['variant'],variant=b['variant'],roots_and_sampled_shapes_equal=True,
        losses_exact=bool(np.array_equal(aa,bb)),max_abs_loss_difference=float(abs(aa-bb).max()),
        final_parameters_exact=a['final_parameters_sha256']==b['final_parameters_sha256'],
        adam_exact=state_hash(left['optimizer'])==state_hash(right['optimizer']),
        loss_model_adam_close=True,rtol=1e-5,atol=1e-6,
        seconds=b['training_seconds'],reference_seconds=a['training_seconds'],speedup=a['training_seconds']/b['training_seconds'])


def verify_version(launch,state,binding,actual):
    require(launch['candidate_sha256']==state['candidate_sha256']==binding['candidate_sha256']==actual,'Wrong actual worker version')


def load_formal(folder):
    from .controller import plan,worker_command,arm_evidence
    from .review import review_worker
    folder=Path(folder);state=read(folder/'status.json');launch=read(folder/'launch.json');binding=read(folder/'inputs.json')
    require(state['schema']=='digit-ig-host-opt-run-v1' and state['passed'] and state['complete'],'Run incomplete')
    require(launch['schema']=='digit-ig-host-opt-launch-v1','Wrong launch')
    verify_version(launch,state,binding,verify())
    require(launch['input_binding_sha256']==sha(folder/'inputs.json') and launch['protocol_sha256']==sha(P) and launch['optimization_protocol_sha256']==sha(HERE/'optimization_protocol.json'),'Protocol/input mismatch')
    require(state['model']==launch['model']=='sage' and state['dataset']==launch['dataset']=='IG' and state['gpu']==launch['gpu'] and state['seed']==launch['seed']==0,'Scope mismatch')
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
    probe=read(folder/'sampling_probe/report.json');receipt=read(folder/'sampling_probe_receipt.json')
    expected_probe=['-B','-u','-m','candidates.ig_sage_host_opt_v1.sampling_probe','--output',str(folder/'sampling_probe'),'--binding',str(folder/'inputs.json')]
    require(receipt['passed'] and receipt['returncode']==0 and receipt['command'][1:]==expected_probe and receipt['report_sha256']==sha(folder/'sampling_probe/report.json') and receipt['log_sha256']==sha(folder/'sampling_probe.log'),'Probe receipt mismatch')
    require(probe['passed'] and probe['candidate_sha256']==verify() and probe['input_binding_sha256']==sha(folder/'inputs.json') and probe['training_updates']==probe['feature_fetches']==0 and not probe['raw_ssd_access'],'Wrong probe scope')
    require(probe['warmup_samples']==20 and probe['measured_samples']==8 and len(probe['native_events'])==len(probe['host_records'])==8,'Incomplete kernel probe')
    for key,raw in [('group_selection_ms','group_sample_ms'),('eid_resolution_ms','resolve_eids_ms'),('native_total_ms','total_ms')]:
        require(probe[key]==sum(r[raw] for r in probe['native_events']),'Kernel aggregate differs')
    variants=['legacy','affinity','combined','compact'];reference=reports['legacy'];comparisons={};rows=[]
    for variant in variants:
        r=reports[variant];comparison=compare_reports(reference,r,folder/'legacy/digit_full',folder/variant/'digit_full')
        if variant!='legacy':require(comparison==read(folder/(variant+'_equivalence.json')),'Equivalence receipt differs')
        comparisons[variant]=comparison;spans={}
        for window in r['stage_profile']['windows']:
            if window['phase']!='training':continue
            for key,value in window['spans'].items():spans[key]=spans.get(key,0.)+value['inclusive_seconds']
        rows.append(dict(variant=variant,seconds=r['training_seconds'],sampling_seconds=spans['sampling'],
            feature_fetch_seconds=spans['feature_fetch'],native_feature_seconds=r['training']['feature_seconds'],
            forward_seconds=spans['forward'],backward_seconds=spans['backward'],adam_seconds=spans['adam'],
            ssd_bytes=r['training']['device']['completed_bytes'],windows=[w['seconds'] for w in r['training']['windows']],
            warmup_seconds=r['warmup_seconds'],setup_seconds=r['setup_seconds'],speedup=comparison['speedup'],
            exact_losses=comparison['losses_exact'],exact_parameters=comparison['final_parameters_exact'],exact_adam=comparison['adam_exact']))
    strict=all(r['external_monitor']['strict_monitor_passed'] for r in reports.values())
    require(strict==state['strict_resource_acceptance'] and state['training_performance_io_passed'],'Final monitor acceptance differs')
    return dict(schema='digit-ig-host-opt-results-v1',passed=True,complete=True,source=str(folder),candidate_sha256=verify(),
        variants=variants,rows=rows,comparisons=comparisons,strict_resource_acceptance=strict,evidence_files_by_worker=counts,
        sampling_probe=probe,one_independent_run_per_variant=True,epoch_time_claim=False,accuracy_claim=False,
        limits=['Training uses unchanged host spans, original warmup/roots/cache/model/Adam and 300 measured batches. No GIDS rerun or change.',
            'Thread affinity and shared-index changes are tested independently and combined; single-run machine variation remains.',
            'Host spans include original waits, not isolated GPU time. Separate sampler-only CUDA events are never added to training E2E.',
            'No validation/test/accuracy/full-epoch claim. Negative results retained. AE unchanged.'])


def emit(value,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write(output/'summary.json',value)
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(value['rows'][0]));writer.writeheader();writer.writerows(value['rows'])
    lines=['# DiGiT IG/SAGE host optimization','',
        'Each variant: 20 warmup + 300 timed batches; one independent process per variant. GIDS and AE unchanged.',
        '', '| Variant | E2E (s) | Sampling (s) | Fetch (s) | Forward (s) | Backward (s) | Adam (s) | Speedup |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        lines.append('| {variant} | {seconds:.2f} | {sampling_seconds:.2f} | {feature_fetch_seconds:.2f} | {forward_seconds:.2f} | {backward_seconds:.2f} | {adam_seconds:.2f} | {speedup:.2f} |'.format(**row))
    p=value['sampling_probe'];lines+=['','Separate 8-batch sampler CUDA-event probe: group %.2f ms, EID %.2f ms, native total %.2f ms. Not training E2E.'%(p['group_selection_ms'],p['eid_resolution_ms'],p['native_total_ms']),'']
    for row in value['rows']:
        lines.append('- %s: all window times %s; exact losses/parameters/Adam: %s/%s/%s.'%(row['variant'],row['windows'],row['exact_losses'],row['exact_parameters'],row['exact_adam']))
    lines+=['',*['- '+v for v in value['limits']]]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
