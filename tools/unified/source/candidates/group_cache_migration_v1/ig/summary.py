"""One matched pair; cache and group denominators stay explicit."""
from .validation import pair_check
def metrics(r):
    t=r['training'];w=t['windows'];total=t['feature_rows'];cpu=t['feature']['cpu'];gpu=sum(x['gpu_hits'] for x in w)
    full=sum(x['mixed'][2] for x in w if x['mixed'] is not None)
    partial=sum(x['mixed'][3] for x in w if x['mixed'] is not None)
    commands=sum(x['primary_commands'] for x in w)
    return dict(seconds=t['seconds'],logical_requests=total,cpu_rows=cpu,gpu_rows=gpu,ssd_rows=total-cpu-gpu,
        cpu_fraction=cpu/total,gpu_fraction=gpu/total,ssd_fraction=(total-cpu-gpu)/total,
        sampled_group_edges=t['group_edges'],sampled_outer_edges=t['outer_edges'],sampled_group_edge_fraction=t['group_edges']/t['outer_edges'],
        ssd=t['device'],primary_8k_commands=full,primary_4k_commands=commands-full,group_companion_4k_commands=partial,
        primary_8k_command_fraction=full/commands if commands else 0,
        windows=w,adjacency_group_edge_fraction=r['group_reuse']['adjacency_group_edge_fraction'])
def comparisons(reports,folder):
    smoke=pair_check({k:reports['smoke_'+k] for k in ('gids','digit_full')},True)
    pair=pair_check(dict(gids=reports['gids_1'],digit_full=reports['digit_1']),False)
    return dict(passed=True,smoke=smoke,pair=pair,aggregation='one pair, no best-of selection',
                arms={k:metrics(reports[v]) for k,v in [('gids','gids_1'),('digit','digit_1')]},
                cache_profile=reports['digit_1']['cache_epoch'],accuracy_evaluated=False)
