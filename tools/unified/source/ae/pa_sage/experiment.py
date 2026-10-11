"""Formal PA paired runs: current-release smoke, external monitoring, full acceptance."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ae.pa_sage.common import ROOT, DEFAULT_DATA, cfg, source_config, read, write, sha, require
from ae.common import verify_release, check_device
from ae.pa_sage.native_gate import current_binding, check_native
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.monitor_validation import monitor_evidence
from ae.pa_sage.full_validation import validate_report, validate_pair, checkpoint_hash

EXECUTION=ROOT/'configs/paper/pa_sage_execution_v4.json'


def make_plan(seeds=None,repeats=None):
    execution=read(EXECUTION);protocol=cfg()
    seeds=execution['seeds'] if seeds is None else seeds
    repeats=execution['repeats'] if repeats is None else repeats
    require(seeds and len(seeds)==len(set(seeds)) and set(seeds)<=set(execution['seeds']), 'Invalid seeds')
    require(repeats and len(repeats)==len(set(repeats)) and set(repeats)<=set(execution['repeats']), 'Invalid repeats')
    require(execution['epochs']==protocol['paper_specified']['epochs']==20,'Wrong experiment length')
    require(execution['seeds']==protocol['reconstruction_choices']['seeds'] and len(execution['repeats'])==protocol['reconstruction_choices']['repeats_per_seed'],'Execution matrix differs from protocol')
    return [dict(seed=s,repeat=r,arm=a,key='seed%d_repeat%d_%s'%(s,r,a))
        for s in seeds for r in repeats for a in (('gids','digit_full') if r%2==0 else ('digit_full','gids'))]



def aggregate_metrics(jobs):
    """Equal weight per seed; repeat variation is retained in per-pair output."""
    import statistics
    fields=('mean_training_epoch_seconds','order_excluded_mean_training_seconds',
            'validation_seconds','test_seconds','trace_preparation_seconds',
            'online_seconds','with_trace_preparation_seconds','test_accuracy')
    by_seed={}
    for seed in sorted({item['metrics']['seed'] for item in jobs.values()}):
        arms={}
        for arm in ('gids','digit_full'):
            values=[item['metrics'] for key,item in jobs.items() if key.endswith('_'+arm) and item['metrics']['seed']==seed]
            require(values,'Missing arm in seed aggregate')
            arms[arm]={field:statistics.mean(v[field] for v in values) for field in fields}
            arms[arm]['repeats']=sorted(v['repeat'] for v in values)
        require(arms['gids']['repeats']==arms['digit_full']['repeats'],'Unpaired repeat aggregate')
        by_seed[str(seed)]=arms
    overall={arm:{field:dict(mean=statistics.mean(v[arm][field] for v in by_seed.values()),
                           sample_std=statistics.stdev(v[arm][field] for v in by_seed.values()) if len(by_seed)>1 else None)
                  for field in fields} for arm in ('gids','digit_full')}
    return dict(by_seed=by_seed,across_seeds=overall,seed_count=len(by_seed),
        policy='Average repeats within each seed, then equal-weight seeds; std is descriptive, not a confidence interval.')


def contract(data,seed,repeat):
    protocol=cfg();splits=source_config()['splits'];traces=read(data/'traces_ready.json')['traces'];orders=read(data/'orders.json')['hashes']
    return dict(seed=seed,repeat=repeat,epochs=20,batch_size=protocol['paper_specified']['batch_size'],
        train_examples=splits['train']['count'],valid_examples=splits['valid']['count'],test_examples=splits['test']['count'],
        classes=protocol['reconstruction_choices']['num_classes'],cpu_cache_rows=protocol['reconstruction_choices']['cpu_cache_rows'],
        root_sha256=[orders['s%d_e%d'%(seed,e)] for e in range(20)],
        valid_trace_sha256=traces['valid']['manifest_sha256'],test_trace_sha256=traces['test']['manifest_sha256'])


def run(data,output,seeds=None,repeats=None):
    require(__debug__ and os.geteuid()==0 and os.environ.get('TMUX'),'Formal runs require sudo inside tmux')
    with open('/tmp/digit-pa-sage-experiment.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(Path(data).resolve(),Path(output).resolve(),make_plan(seeds,repeats))


def _run(data,output,plan):
    output.mkdir(parents=True,exist_ok=False)
    status=dict(schema='digit-pa-formal-v4',stage='preflight',passed=False,complete=False,pid=os.getpid(),
        started_unix=time.time(),plan=plan,workers=[],raw_writes=False)
    def save():status['updated_unix']=time.time();write(output/'status.json',status)
    save();child=None;monitor=ExternalMonitor(output/'external_gpu')
    try:
        release=verify_release();binding=current_binding(data);check_device()
        native=check_native(data)
        if not native['passed']:
            from ae.pa_sage.native_smoke import run as smoke
            status['stage']='current_release_native_smoke';save()
            smoke(data,output/'native_smoke')
            native=check_native(data)
        require(native['passed'],'New-release native smoke did not pass')
        require(verify_release()==release and current_binding(data)==binding,'Source/input binding changed in preflight')
        launch=dict(execution_sha256=release,execution_profile_sha256=sha(EXECUTION),binding=binding,
            native_receipt_sha256=native['receipt_sha256'],plan=plan,contracts={job['key']:contract(data,job['seed'],job['repeat']) for job in plan})
        write(output/'launch_manifest.json',launch)
        status['external_monitor']=monitor.start();save()
        reports={}
        for job in plan:
            check_device();folder=output/job['key']
            cmd=[sys.executable,'-u',str(ROOT/'ae/run_pa_sage.py'),'train','--data',str(data),
                '--arm',job['arm'],'--seed',str(job['seed']),'--repeat',str(job['repeat']),'--output',str(folder/'report.json')]
            worker=dict(job,command=cmd,status='running',started_unix=time.time())
            status['workers'].append(worker);status['stage']=job['key']
            with (output/(job['key']+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                worker['pid']=child.pid;save();code=child.wait();child=None
            worker.update(returncode=code,finished_unix=time.time(),status='exited' if code==0 else 'failed');save()
            require(code==0,'Worker failed: '+job['key'])
            report=read(folder/'report.json')
            metrics=validate_report(report,job['arm'],binding,launch['contracts'][job['key']],native['receipt_sha256'])
            require(checkpoint_hash(folder/'final_model.pt')==report['final_parameters_sha256'],'Saved checkpoint mismatch')
            reports[job['key']]=report
            write(folder/'accepted_functional.json',dict(passed=True,metrics=metrics,report_sha256=sha(folder/'report.json'),checkpoint_sha256=sha(folder/'final_model.pt')))
        status['external_monitor_returncode']=monitor.stop();save()
        require(status['external_monitor_returncode']==0,'External monitor failed')
        check_device();require(verify_release()==release and current_binding(data)==binding,'Frozen execution changed during experiment')
        metrics={}
        for job in plan:
            folder=output/job['key'];item=read(folder/'accepted_functional.json')
            # Select by unique job key: the full matrix repeats each arm.
            one=dict(status,workers=[w for w in status['workers'] if w['key']==job['key']])
            item['external_monitor']=monitor_evidence(output,one,job['arm'],read(folder/'resources.json'))
            require(item['external_monitor']['observed_peak_device_used_bytes']<=reports[job['key']]['admission']['required_bytes'],'Observed GPU use exceeded admission')
            require(item['metrics']['peak_host_rss_bytes']<=reports[job['key']]['admission']['host_required_bytes'],'Observed host use exceeded admission')
            item['resources_sha256']=sha(folder/'resources.json');write(folder/'accepted.json',item)
            metrics[job['key']]=item
        pairs=[]
        for seed,repeat in dict.fromkeys((j['seed'],j['repeat']) for j in plan):
            pair={a:reports['seed%d_repeat%d_%s'%(seed,repeat,a)] for a in ('gids','digit_full')}
            pairs.append(dict(seed=seed,repeat=repeat,**validate_pair(pair)))
        summary=dict(passed=True,complete=True,execution_sha256=release,workers=metrics,pairs=pairs,aggregates=aggregate_metrics(metrics),
            scope='Requested PA/GraphSAGE subset; average repeats within seed before cross-seed summaries.',
            external_monitor_hashes={name:sha(output/'external_gpu'/name) for name in ('samples.jsonl','summary.json','ready.json')})
        write(output/'summary.json',summary)
        status.update(stage='complete',passed=True,complete=True,finished_unix=time.time());save()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        monitor.stop();status.update(stage='failed',error=type(exc).__name__+': '+str(exc),finished_unix=time.time());save();raise
