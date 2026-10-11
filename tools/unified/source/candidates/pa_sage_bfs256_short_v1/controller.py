"""Serial bounded preparation/profile/smoke/pair; no automatic failure retry."""
import argparse
import fcntl
import signal
import subprocess
import time
from contextlib import ExitStack
from .common import *

def aggregate(reports):
    require(set(reports)=={'gids','digit'},'Missing arm')
    for arm,r in reports.items():
        require(r['passed'] and not r['smoke'] and r['arm']==arm and r['updates']==320 and r['measured_batches']==300,'Incomplete arm')
        require(r['training']['reconciled'] and r['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'Counters/cold cache failed')
        require(r['protocol']==cfg() and r['source_sha256']==verify(),'Wrong configuration')
    a,b=reports['gids'],reports['digit']
    require(a['initial_model_sha256']==b['initial_model_sha256'],'Unmatched initial model')
    from .dataset import check_report_roots,comparison_roots
    for arm,r in reports.items():check_report_roots(r,arm)
    root_comparison=comparison_roots()
    for k in ('cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes'):require(a['cache'][k]==b['cache'][k],'Unequal feature cache capacity')
    require(b['pipeline']['submitted']==b['pipeline']['collected']==320 and b['pipeline']['outstanding']==0,'Incomplete async pipeline')
    rows={}
    for arm,r in reports.items():
        windows=r['windows'][1:];groups=sum(w['group_sampled_edges'] for w in windows);edges=sum(w['outer_sampled_edges'] for w in windows)
        ssd=r['training']['serving']['ssd_served_rows'];pairs=r['training']['request_sizes']['primary_1k']
        rows[arm]=dict(seconds=r['measured_seconds'],serving=r['training']['serving'],device=r['training']['device'],
            request_sizes=r['training']['request_sizes'],group_sampled_edge_fraction=groups/edges if edges else 0.,
            ssd_grouped_row_fraction=2*pairs/ssd if ssd else 0.,setup_seconds=r['setup_seconds'])
    return dict(passed=True,complete=True,arms=rows,speedup=a['measured_seconds']/b['measured_seconds'],
        root_comparison=root_comparison,aggregation='one random-GIDS versus BFS-DiGiT pair, all 300 timed batches retained',accuracy_evaluated=False,
        stable_speedup_established=False,full_epoch_training=False,
        limitations=['Different root orders and short-window examples; system comparison includes ordering, cache, sampling and execution differences.','Actual PA features/labels but no validation/test in this short pair.',
                    'One short comparison does not establish paper performance or accuracy.'])
