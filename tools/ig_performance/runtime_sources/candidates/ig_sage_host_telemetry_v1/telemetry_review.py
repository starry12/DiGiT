"""Align sampled scheduling and clock observations to original timed windows."""
import json,statistics
from pathlib import Path
from .common import require,read


def stats(values):
    values=[x for x in values if x is not None]
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values),mean=statistics.mean(values)) if values else dict(count=0,minimum=None,median=None,maximum=None,mean=None)


def midpoint(v):return (v['start_perf_ns']+v['end_perf_ns'])//2


def coverage(records,start,end,max_gap,minimum=2):
    points=[v for v in records if v.get('ok') and start<=v['start_perf_ns']<=v['end_perf_ns']<=end]
    times=[start]+[midpoint(v) for v in points]+[end]
    gap=max(b-a for a,b in zip(times,times[1:]))/1e9
    return points,dict(samples=len(points),max_gap_seconds=gap,complete=len(points)>=minimum and gap<=max_gap)


def thread_delta(samples,start,end,pid):
    good=[v for v in samples if v.get('ok') and str(pid) in v['threads']]
    left=[v for v in good if v['end_perf_ns']<=start];right=[v for v in good if v['start_perf_ns']>=end]
    if not left or not right:return dict(available=False,reason='Missing CPU samples bracketing the timed window')
    a,b=left[-1],right[0];x,y=[v['threads'][str(pid)] for v in (a,b)]
    if x['starttime_ticks']!=y['starttime_ticks']:return dict(available=False,reason='PID identity changed')
    def delta(k):return y[k]-x[k] if x[k] is not None and y[k] is not None else None
    names=('utime_ticks','stime_ticks','runtime_ns','voluntary_switches','involuntary_switches','runqueue_wait_ns','timeslices')
    differences={k:delta(k) for k in names}
    if any(v is not None and v<0 for v in differences.values()):return dict(available=False,reason='CPU counters decreased')
    per_cpu={}
    for cpu in map(str,a['cpu2_smt_siblings']):
        aa,bb=a['cpu_ticks'].get(cpu),b['cpu_ticks'].get(cpu)
        if aa is None or bb is None:continue
        d=[yy-xx for xx,yy in zip(aa,bb)];total=sum(d)
        per_cpu[cpu]=dict(busy_percent=100*(total-d[3]-d[4])/total if total>0 else None,
                           iowait_percent=100*d[4]/total if total>0 else None)
    seq=[v['threads'][str(pid)]['last_cpu'] for v in good if a['start_perf_ns']<=v['start_perf_ns']<=b['start_perf_ns']]
    return dict(available=True,bracket_start_perf_ns=a['start_perf_ns'],bracket_end_perf_ns=b['end_perf_ns'],
        bracket_seconds=(b['end_perf_ns']-a['start_perf_ns'])/1e9,
        outside_window_seconds=((start-a['start_perf_ns'])+(b['end_perf_ns']-end))/1e9,
        main_thread=differences,observed_cpu_ids=sorted(set(seq)),
        sampled_cpu_changes_lower_bound=sum(x!=y for x,y in zip(seq,seq[1:])),cpu2_and_smt_utilization=per_cpu,
        runqueue_wait_available=all(v['schedstats_enabled'] for v in (a,b)))


def review(folder,r):
    folder=Path(folder);ready=read(folder/'ready.json');summary=read(folder/'summary.json');exit_=read(folder/'exit.json')
    require(ready['passed'] and ready['candidate_sha256']==summary['candidate_sha256']==r['candidate_sha256'],'Wrong telemetry version')
    pid=r['resource_observations']['worker_pid']
    require(ready['target_pid']==summary['target_pid']==pid,'Wrong telemetry worker')
    require(ready['monitor_cpu']==summary['monitor_cpu'] and ready['monitor_cpu'] not in ready['excluded_cpus'] and 2 in ready['excluded_cpus'],'Telemetry shares CPU2 or its SMT sibling')
    require(ready['gpu_uuid']=='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703','Wrong telemetry GPU')
    def lines(name):return [json.loads(line) for line in (folder/name).read_text().splitlines()]
    cpu,gpu,markers=lines('cpu.jsonl'),lines('gpu.jsonl'),lines('windows.jsonl')
    expected=[(name,w) for name in ('warmup','training') if r[name] for w in r[name]['windows']]
    require(len(markers)==len(expected),'Incomplete telemetry window markers')
    require(len(cpu)==summary['samples'],'CPU telemetry count differs')
    for source in (cpu,gpu):
        require(all(v['sequence']==i and v['start_perf_ns']<=v['end_perf_ns'] for i,v in enumerate(source)),'Invalid telemetry sequence/timestamp')
        require(all(a['end_perf_ns']<=b['start_perf_ns'] for a,b in zip(source,source[1:])),'Telemetry timestamps overlap')
    output=[]
    for marker,(phase,w) in zip(markers,expected):
        require(marker['phase']==phase and marker['index']==w['index'] and marker['seconds']==w['seconds'],'Telemetry/profile window differs')
        start,end=marker['start_perf_ns'],marker['end_perf_ns']
        require(abs((end-start)/1e9-w['seconds'])<1e-6,'Telemetry duration differs')
        cc,ccover=coverage(cpu,start,end,1.0);gg,gcover=coverage(gpu,start,end,2.0)
        available=sorted({k for v in cc for k in v['cpu_frequency_khz']})
        frequencies={k:stats([v['cpu_frequency_khz'].get(k) for v in cc]) for k in available}
        scheduling=thread_delta(cpu,start,end,pid)
        observed_freq=[v['cpu_frequency_khz'].get(str(t['last_cpu'])) for v in cc for t in v['threads'].values()]
        rows=dict(phase=phase,index=w['index'],seconds=w['seconds'],start_perf_ns=start,end_perf_ns=end,
            cpu_coverage=ccover,gpu_coverage=gcover,scheduling=scheduling,cpu_frequency_khz=frequencies,
            observed_worker_cpu_frequency_khz=stats(observed_freq),
            main_thread_cpu_frequency_khz=stats([v['cpu_frequency_khz'].get(str(v['threads'][str(pid)]['last_cpu'])) for v in cc]),
            gpu={k:stats([v.get(k) for v in gg]) for k in ('graphics_mhz','sm_mhz','memory_mhz','power_mw','temperature_c')},
            gpu_pstates=sorted({v['pstate'] for v in gg if v.get('pstate') is not None}),
            gpu_throttle_reason_masks=sorted({v['throttle_reasons'] for v in gg if v.get('throttle_reasons') is not None}),
            cpu_sample_read_errors=[v['thread_read_errors'] for v in cc if v['thread_read_errors']],
            gpu_query_errors=[v for v in gpu if not v.get('ok') and start<=midpoint(v)<=end])
        rows['diagnostic_complete']=ccover['complete'] and gcover['complete'] and scheduling['available'] and rows['observed_worker_cpu_frequency_khz']['count']>0
        output.append(rows)
    gpu_exit=read(folder/'gpu_exit.json') if (folder/'gpu_exit.json').exists() else {}
    complete=summary['complete'] and summary['stopped_by_worker'] and exit_['returncode']==0 and summary['gpu_returncode']==0 and all(w['diagnostic_complete'] for w in output)
    return dict(schema='ig-training-telemetry-review-v1',diagnostic_complete=complete,windows=output,
        monitor_cpu=ready['monitor_cpu'],schedstats_enabled=ready['schedstats_enabled'],
        collector_cpu_seconds=summary['cpu_seconds'],gpu_collector_cpu_seconds=gpu_exit.get('cpu_seconds'),
        collector_wall_seconds=summary['wall_seconds'],cpu_samples=len(cpu),gpu_samples=len(gpu),
        cpu_sample_duration_seconds=stats([(v['end_perf_ns']-v['start_perf_ns'])/1e9 for v in cpu]),
        gpu_query_duration_seconds=stats([(v['end_perf_ns']-v['start_perf_ns'])/1e9 for v in gpu]),
        limits=['CPU and GPU clocks are sampled, not continuous or cycle-weighted traces.',
            'Context-switch deltas bracket each timed window with up to a sampling-period margin; voluntary switches include normal waits.',
            'Observed CPU migrations are lower bounds, not a complete scheduler trace. Main-thread scheduling is summarized; raw records retain every observed worker thread.',
            'If kernel sched_schedstats=0, runqueue wait is unavailable, not zero. No global setting was changed.',
            'GPU throttle masks include idle/application states; a nonzero mask alone does not prove performance throttling.',
            'Extra recorder CPU use is reported. It does not measure or subtract its exact effect on training E2E.'])
