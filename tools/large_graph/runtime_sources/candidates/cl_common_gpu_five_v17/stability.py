"""Summarize actual accepted cycles separately from the best speedup."""
import hashlib
import json
import statistics
from pathlib import Path
from .numa import valid_report


def review(series):
    if not (series['passed'] is True and series['runs_complete']==10 and series['rounds_complete']==5):
        raise ValueError('Five complete accepted pairs required for stability evidence')
    runs=[]
    for pair in series['pairs']:
        for arm in ('gids','digit'):
            ref=pair[arm];p=Path(ref['path'])
            if hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:
                raise RuntimeError('Acceptance changed before stability review')
            a=json.loads(p.read_text());w=a['worker']
            if not (a['passed'] is True and a['pair_id']==pair['pair_id']
                    and a['gpu_released'] and a['post_guard_passed'] and a['kernel_monitor_ok']
                    and valid_report(w['numa_placement'],arm)):
                raise RuntimeError('Incomplete lifecycle or placement evidence')
            first=last=None;peak_full=peak_cg=0;peak_nodes={};samples=diagnostics=0
            with (p.parent/'monitor.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row.get('phase')!='runtime':continue
                    s=row['sample'];samples+=1
                    pressure=s['psi']['memory']['full']
                    if first is None:first=pressure['total']
                    last=pressure['total'];peak_full=max(peak_full,pressure['avg10'])
                    peak_cg=max(peak_cg,s.get('cgroup',{}).get('current',0))
                    d=s.get('numa_diagnostics',{})
                    if d.get('ok'):diagnostics+=1
                    for n,v in d.get('cgroups',{}).get('service',{}).get('numa',{}).get('anon',{}).items():
                        peak_nodes[n]=max(peak_nodes.get(n,0),int(v))
            runs.append(dict(round=pair['round'],arm=arm,seconds=w['seconds'],acceptance=str(p),
                numa_placement=w['numa_placement'],runtime_samples=samples,numa_diagnostic_samples=diagnostics,
                memory_full_stall_seconds=(last-first)/1e6 if first is not None else None,
                memory_full_peak_avg10=peak_full,cgroup_peak_bytes=peak_cg,cgroup_peak_anon_by_node=peak_nodes))
    timings={}
    for arm in ('gids','digit'):
        values=[r['seconds'] for r in runs if r['arm']==arm]
        timings[arm]=dict(min=min(values),max=max(values),mean=statistics.mean(values),
            population_stdev=statistics.pstdev(values),relative_range=(max(values)-min(values))/statistics.mean(values))
    return dict(five_pairs_accepted=True,digit_interleaved_cycles=5,runs=runs,timing_variation=timings,
        note='Five accepted allocation/training/release cycles; no guarantee of all future runs or timing invariance.')
