"""Lightweight plans: no graph scans, GPU imports or device initialization."""
from pathlib import Path
from candidates.unified_training_v2.config import MODES,load_config
from .common import ROOT
DATASETS=('PA','IG','UKS','UKL','CL')
BACKENDS={'PA':'pa_sage_bfs256','IG':'ig_unified_v2','UKS':'uks_unified_v2','UKL':'ukl_unified_v2','CL':'cl_unified_v2'}
def runtime_module(dataset,mode):
    """Single dispatch source for both preview and execution; no backend import."""
    if dataset not in DATASETS or mode not in MODES:raise ValueError('Unknown dataset/mode')
    if dataset=='PA':
        return ('candidates.pa_sage_bfs256_full20_v1.controller' if mode=='full20'
                else 'candidates.unified_experiments_v2.pa_short')
    suffix={'IG':'launcher','UKS':'controller','UKL':'runtime.start','CL':'runtime.start'}[dataset]
    return 'candidates.'+BACKENDS[dataset]+'.'+suffix

def plan(dataset,mode):
    if dataset not in DATASETS or mode not in MODES:raise ValueError('Unknown dataset/mode')
    c=load_config(dataset,mode)
    order='GIDS_random_DiGiT_BFS256'
    if dataset=='PA':
        from dataclasses import asdict
        c=dict(schema='digit-pa-existing-protocol-entry-v1',dataset=dataset,
            execution=asdict(MODES[mode]),
            root_order_comparison=order,row_bytes=512,accuracy_evaluated=mode=='full20',
            protocol_source='candidates.pa_sage_bfs256_full20_v1.common.cfg' if mode=='full20' else 'candidates.pa_sage_bfs256_short_v1.common.cfg')
    else:
        c['execution_capabilities']=dict(digit_async_prefetch=dataset in ('IG','UKS'),digit_model_optimization=True,gpu_capture=True)
        if dataset in ('UKS','UKL','CL'):
            from importlib import import_module
            c=import_module('candidates.'+BACKENDS[dataset]+'.common').config(mode)
        elif dataset=='IG':c=load_config(dataset,mode)
    files=[]
    if dataset=='UKL':files=['/mnt/n0/digit/ukl_compact_cover_20261011_v1/covered.u32.json']
    if dataset=='CL':files=['/mnt/n0/digit/group_cache_migration_v1/CL/prepared.json']
    return dict(dataset=dataset,mode=mode,backend=BACKENDS[dataset],runtime_module=runtime_module(dataset,mode),config=c,
        schedule=[dict(round=r,arm=a) for r,a in MODES[mode].schedule()],root_order_comparison=order,
        missing_preparation=[p for p in files if not Path(p).is_file()],
        preparation_during_run='IG builds BFS and profiles if needed; UKL/CL require separately accepted preparation',
        aggregation='median_of_five_paired_speedups' if mode=='short5' else 'single_pair_ratio_of_total_training_time',
        accuracy_evaluation=dataset=='PA' and mode=='full20',
        native_acceptance_claimed=False,wait_poll_seconds=60,raw_ssd_writes=False)
