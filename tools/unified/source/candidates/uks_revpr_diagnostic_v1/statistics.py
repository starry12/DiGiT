import math
from .common import HERE,read,require
STAGES=('root_transfer','sampling','address_annotation','feature_fetch','label_transfer','forward','loss','zero_grad','backward','adam')
def jobs():return read(HERE/'protocol.json')['schedule']
def summarize(rows):
    require([(r['mode'],r['variant'],r['repetition'],r['profile_mode']) for r in rows]==[(j['mode'],j['variant'],j['repetition'],j['profile_mode']) for j in jobs()],'Missing or reordered runs')
    for r in rows:
        require(r['passed'] and r['normal_exit'] and r['updates']==320 and r['measured_batches']==300 and r['warmup_batches']==20,'Incomplete run')
        require(len(r['windows_seconds'])==3 and all(math.isfinite(t) and t>0 for t in r['windows_seconds']) and math.isclose(sum(r['windows_seconds']),r['seconds'],rel_tol=1e-12),'Bad timing')
    for key in ('roots_sha256','initial_model_sha256','hot_file_sha256'):require(len({r[key] for r in rows})==1,'Unpaired '+key)
    require(len({r['cache']['logical_hot_sha256'] for r in rows})==1,'Different installed CPU hot set')
    perf=[r for r in rows if r['profile_mode']=='off'];profiles={}
    comparisons={}
    for variant in ('digit_default','digit_cpu2'):
        pairs=[]
        for i in range(1,6):
            pair={r['variant']:r for r in perf if r['repetition']==i};g=pair['gids_default']['seconds'];d=pair[variant]['seconds']
            pairs.append(dict(round=i,gids_seconds=g,digit_seconds=d,speedup=g/d))
        best=max(pairs,key=lambda p:p['speedup']);comparisons[variant]=dict(paired_rounds=pairs,selected_round=best['round'],max_observed_speedup=best['speedup'])
    affinity=[]
    for i in range(1,6):
        p={r['variant']:r for r in perf if r['repetition']==i};affinity.append(dict(round=i,default_seconds=p['digit_default']['seconds'],cpu2_seconds=p['digit_cpu2']['seconds'],cpu2_speedup=p['digit_default']['seconds']/p['digit_cpu2']['seconds']))
    for r in rows:
        if r['profile_mode']!='stages':continue
        totals={k:dict(seconds=0.,calls=0) for k in STAGES}
        for w in r['windows'][1:]:
            require(set(w['stages'])==set(STAGES),'Missing diagnostic stage')
            for k,v in w['stages'].items():
                require(v['calls']==100 and math.isfinite(v['seconds']) and v['seconds']>=0,'Incomplete stage')
                totals[k]['calls']+=v['calls'];totals[k]['seconds']+=v['seconds']
        require(sum(v['seconds'] for v in totals.values())<=r['seconds']+1e-6,'Double counted stage time')
        profiles[r['variant']]=dict(stages=totals,instrumented_seconds=r['seconds'],unattributed_seconds=r['seconds']-sum(v['seconds'] for v in totals.values()))
    return dict(passed=True,comparisons=comparisons,affinity_pairs=affinity,profiles=profiles,full_epoch=False,accuracy_claim=False,selection_rule='maximum_same_round_speedup_out_of_five',profile_times_used_for_speedup=False)
