"""Keep original strict-monitor failure; independently qualify sparse timeout gaps."""
import copy,math
from evaluation.sage.common import *
from training.sage.validation import report_check,pair_check
from training.sage.worker import check_inputs
from runtime.io.accounting import summarize

def assess_monitor(summary,records,ready,worker,controller_pid,resources,returncode,gpu):
    require(returncode==0 and summary['complete'],'Monitor did not finish normally')
    require(ready['passed'] and summary['pid']==ready['pid'] and summary['parent_pid']==controller_pid,'Wrong monitor ownership')
    require(summary['gpu']==str(gpu) and summary['mode']=='external_small_process','Wrong GPU/monitor')
    require(summary['pid'] not in (worker['pid'],controller_pid),'Monitor not independent')
    require(resources['worker_pid']==worker['pid'] and resources['mode']=='checkpoints_only_external_sampler' and not resources['background_monitor_in_worker'] and resources['monitor_samples']==0,'Wrong worker resources')
    require(summary['peak_rss_bytes']<=64*2**20,'Monitor RSS exceeds budget')
    require(records and all(math.isfinite(x['time_unix']) for x in records),'Missing/invalid samples')
    require(all(a['time_unix']<b['time_unix'] for a,b in zip(records,records[1:])),'Nonmonotonic monitor records')
    errors=[x for x in records if 'error' in x];good=[x for x in records if 'error' not in x]
    require(errors==summary['errors'] and len(good)==summary['samples'],'Monitor summary differs from raw records')
    require(summary['passed']==(len(good)>0 and not errors),'Wrong original monitor acceptance')
    require(ready['first_sample']==good[0],'Readiness sample differs')
    command=['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']
    expected="TimeoutExpired: Command '%s' timed out after 5 seconds"%command
    require(all(x['error']==expected for x in errors),'Non-timeout monitoring error')
    require(len(errors)/len(records)<=POLICY['max_error_fraction'],'Too many monitoring errors')
    consecutive=maximum=0
    for rec in records:
        consecutive=consecutive+1 if 'error' in rec else 0;maximum=max(maximum,consecutive)
    require(maximum<=POLICY['max_consecutive_errors'],'Consecutive monitoring failures')
    require(all(x['monitor_pid']==summary['pid'] and x['monitor_parent_pid']==controller_pid for x in good),'Wrong sample identity')
    require(all(isinstance(x['device_used_bytes'],int) and x['device_used_bytes']>=0 and 0<=x['utilization_percent']<=100 and 0<=x['monitor_rss_bytes']<=64*2**20 for x in good),'Invalid sample metrics')
    require(max(x['device_used_bytes'] for x in good)==summary['peak_device_used_bytes'],'Monitor peak differs')
    gaps=[b['time_unix']-a['time_unix'] for a,b in zip(good,good[1:])]
    gap=max(gaps,default=math.inf);require(gap<=POLICY['max_sample_gap_seconds'],'Excessive monitoring gap')
    points=[x for x in good if worker['started_unix']<=x['time_unix']<=worker['finished_unix']]
    require(len(points)>=10,'Insufficient worker samples')
    checkpoints=resources['checkpoints'];require(checkpoints and checkpoints[0]['stage']=='start' and checkpoints[-1]['stage']=='training_complete','Incomplete worker checkpoints')
    require(all(worker['started_unix']<=x['time_unix']<=worker['finished_unix'] for x in checkpoints),'Checkpoints outside worker lifetime')
    require(good[0]['time_unix']<=checkpoints[0]['time_unix']+POLICY['max_endpoint_gap_seconds'] and good[-1]['time_unix']>=checkpoints[-1]['time_unix']-POLICY['max_endpoint_gap_seconds'],'Monitor does not cover training lifetime')
    require(max(x['device_used_bytes'] for x in checkpoints)==resources['observed_peak_device_used_bytes'],'Checkpoint peak differs')
    peak=max(resources['observed_peak_device_used_bytes'],max(x['device_used_bytes'] for x in points))
    return dict(monitor_pid=summary['pid'],worker_pid=worker['pid'],sample_count=len(points),monitor_peak_rss_bytes=summary['peak_rss_bytes'],observed_peak_device_used_bytes=peak,
                strict_monitor_passed=summary['passed'],qualification='observed_with_sparse_query_timeouts' if errors else 'no_query_errors',
                timeout_count=len(errors),query_error_fraction=len(errors)/len(records),max_sample_gap_seconds=gap,
                review_policy=POLICY,limitation='Sampled peak is a lower bound; missing samples cannot establish memory peaks during gaps. This review does not convert the original strict monitor failure into a pass.')

def review_worker(base,worker,controller_pid,binding,gpu):
    arm=worker['arm'];mode=worker['mode'];folder=base/mode/arm;mon=base/(mode+'_monitor_'+arm)/'external_gpu'
    require(worker['status']=='complete' and worker['returncode']==0,'Worker did not exit normally')
    require(read(folder/'worker_ready.json')['passed'] and sha(folder/'report.json')==read(folder/'worker_ready.json')['report_sha256'],'Worker report incomplete/changed')
    release=read(folder/'release_worker.json');require(release['monitor_stopped'],'Monitor not stopped')
    r=read(folder/'report.json');resources=read(folder/'resources.json');require(resources==r['resource_observations'],'Resource report differs')
    records=[__import__('json').loads(line) for line in (mon/'samples.jsonl').read_text().splitlines()]
    r['external_monitor']=assess_monitor(read(mon/'summary.json'),records,read(mon/'ready.json'),worker,controller_pid,resources,release['monitor_returncode'],gpu)
    require(r['input_binding_sha256']==sha(base/'inputs.json'),'Wrong input binding hash')
    if mode=='smoke':require(r['external_monitor']['strict_monitor_passed'],'Smoke requires an error-free monitor')
    report_check(r,arm,mode=='smoke',binding,folder)
    require(r['peak_host_rss_kib']*1024<=r['admission']['host_required_bytes'],'Observed host memory exceeds budget')
    for e in r['epochs']:require(read(folder/('epoch_%02d.json'%e['epoch']))==e,'Epoch report changed')
    def phase(regions,name):
        value=summarize(regions);value['phase']=name;value['evaluation_seconds']=value.pop('train_seconds');value['ssd_useful_per_evaluation_gbps']=value.pop('ssd_useful_per_train_gbps');return value
    require(r['io_accounting_validation']==phase([(e['validation'],e['validation']['seconds']) for e in r['epochs']],'validation'),'Validation I/O aggregation differs')
    if mode=='full':
        require(r['test']['useful_io']['region_id']==41,'Test counter phase leakage')
        require(r['io_accounting_test']==phase([(r['test'],r['test']['seconds'])],'test'),'Test I/O aggregation differs')
    r['submission_review']=dict(training_accuracy_and_io_passed=True,original_strict_monitor_passed=r['external_monitor']['strict_monitor_passed'],original_report=str((folder/'report.json').relative_to(ROOT)),original_report_sha256=sha(folder/'report.json'),posthoc_monitor_review=False)
    return r
