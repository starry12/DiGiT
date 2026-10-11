import json
from .common import *
from candidates.ig_demand_pair_v1.monitor_review import assess_monitor
from candidates.unified_training_v2.accounting import validate_counts
def review_worker(base,worker,controller_pid,binding,gpu):
    folder=base/worker['mode']/worker['arm'];mon=base/(worker['mode']+'_monitor_'+worker['arm'])/'external_gpu'
    require(worker['status']=='complete' and worker['returncode']==0,'Abnormal worker exit')
    ready=read(folder/'worker_ready.json');require(ready['passed'] and ready['report_sha256']==sha(folder/'report.json'),'Incomplete/changed report')
    release=read(folder/'release_worker.json');require(release['monitor_stopped'],'Monitor not stopped')
    r=read(folder/'report.json');resources=read(folder/'resources.json')
    require(resources==r['resource_observations'] and r['input_binding_sha256']==sha(base/'launch.json'),'Resource/input evidence differs')
    records=[json.loads(line) for line in (mon/'samples.jsonl').read_text().splitlines()]
    r['external_monitor']=assess_monitor(read(mon/'summary.json'),records,read(mon/'ready.json'),worker,controller_pid,resources,release['monitor_returncode'],gpu)
    require(r['external_monitor']['strict_monitor_passed'],'Strict external monitoring required')
    require(r['passed'] and r['complete'] and r['initial']['source_sha256']==verify(),'Unaccepted native report')
    if worker['smoke']:
        require(r['smoke'] and len(r['audits'])==2 and all(x['source_features_equal'] and x['original_edges_verified'] for x in r['audits']),'Missing source-feature/edge smoke')
        validate_counts(r['window']['counters'],4096,'digit' if worker['arm']=='digit_full' else 'gids')
    else:
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        epochs=[{k:v for k,v in e.items() if k!='stage'} for e in events if e['stage']=='epoch_complete']
        require(epochs==r['epochs'],'Raw epoch records differ')
        require(r['repetition']==worker['repetition'] and r['mode']==worker['scope'],'Wrong run scope')
    from candidates.uks_group_first_cpu2_v1.resources import target_gpu_busy
    require(not target_gpu_busy(),'GPU2 not released after native worker exit')
    return r
