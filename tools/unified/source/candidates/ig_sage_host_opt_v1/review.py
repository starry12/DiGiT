from candidates.ig_sage_host_opt_v1.common import *
from candidates.ig_sage_host_opt_v1.validation import report_check,pair_check
from candidates.ig_monitor_v1.review import assess_monitor
def review_worker(base,worker,controller_pid,binding,gpu):
    setup();arm=worker['arm'];mode=worker['mode'];folder=base/mode/arm;mon=base/(mode+'_monitor_'+arm)/'external_gpu'
    require(worker['status']=='complete' and worker['returncode']==0,'Abnormal IG worker exit')
    ready=read(folder/'worker_ready.json');require(ready['passed'] and ready['report_sha256']==sha(folder/'report.json'),'Incomplete/changed report')
    release=read(folder/'release_worker.json');require(release['monitor_stopped'],'Monitor still running')
    r=read(folder/'report.json');require(r['model_name']==worker['model'],'Worker model differs');resources=read(folder/'resources.json');require(resources==r['resource_observations'],'Resource report differs')
    records=[json.loads(line) for line in (mon/'samples.jsonl').read_text().splitlines()]
    r['external_monitor']=assess_monitor(read(mon/'summary.json'),records,read(mon/'ready.json'),worker,controller_pid,resources,release['monitor_returncode'],gpu)
    require(r['input_binding_sha256']==sha(base/'inputs.json'),'Wrong input binding')
    if worker['smoke']:require(r['external_monitor']['strict_monitor_passed'],'Smoke requires zero monitor errors')
    require(r['variant']==worker['variant'],'Wrong optimization variant')
    bound=worker['variant'] in ('affinity','combined')
    masks=r['affinity']['final']['thread_masks']
    require(bool(masks),'No thread affinity evidence')
    require(r['affinity']['initial']['policy']==('one_core' if bound else 'unchanged'),'Wrong CPU policy')
    if bound:require(all(mask==[2] for mask in masks.values()),'Worker threads escaped CPU2')
    report_check(r,arm,worker['smoke'],binding,folder)
    from candidates.ig_sage_host_opt_v1.profile import validate
    require(r['stage_profile']==read(folder/'stage_profile.json'),'Raw profile differs')
    require(r['stage_profile']['mode']==worker['profile_mode'],'Wrong profile mode')
    validate(r)
    # Raw windows independently reproduce phase aggregates and cover every batch.
    from candidates.ig_sage_host_opt_v1.features import aggregate
    from candidates.io_accounting_v1.accounting import COUNTERS
    windows=[json.loads(line) for line in (folder/'io_windows.jsonl').read_text().splitlines()]
    from candidates.ig_sage_host_opt_v1.windows import root_slices
    ranges=root_slices(cfg(),worker['smoke'])
    require(all(w['phase'] in [x[0] for x in ranges] for w in windows),'Unexpected I/O phase')
    for phase,lo,hi,width in ranges:
        seq=[{k:v for k,v in w.items() if k!='phase'} for w in windows if w['phase']==phase];v=r[phase]
        require(seq==v['windows'],'Raw/report timing windows differ')
        total=aggregate(seq);require(all(v[k]==x for k,x in total.items()),'Window/phase I/O differs')
    return r
