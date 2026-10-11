"""Freeze the approved comparison plus its explicit, deferred execution sequence."""
from candidates.pa_sage_cache_policy_v1.protocol import compile_protocol, validate as validate_base
from .common import ROOT, HERE, LARGE, GPU, GPU_UUID, ARMS, read, require, sha


def compile_plan():
    p = compile_protocol()
    train_count=read(ROOT/'configs/papers/protocol.json')['dataset']['splits']['train']['count']
    p['execution'] = dict(schema='digit-cache-controller-v3', backend='pa_sage_cache_policy_v2',
        backend_source_sha256=sha(ROOT / 'candidates/pa_sage_cache_policy_v2/manifest.json'),
        large_output_root=str(LARGE), physical_gpu=GPU, gpu_uuid=GPU_UUID,
        monitor='nvidia-smi', monitor_query_timeout_seconds=5, monitor_period_after_query_seconds=.5,
        monitor_max_gap_enforced=False, smoke_updates=4, window_updates=100,
        formal_updates=1179, training_examples=train_count, stages=['bind', 'build', 'rank', 'profile', 'prepare', 'smoke', 'full', 'aggregate'],
        all_smokes_before_any_full=True, fresh_process_per_arm=True,
        full_trace='two position-weighted int64 GPU reductions per sampled tensor; CPU SHA256 after epoch',
        trace_overhead='included equally in each measured epoch, not an exact cryptographic proof of every GPU tensor',
        raw_ssd_writes=False, automatic_start=False)
    p['implementation'] = dict(controller_source_ready=True, native_ready=False,
        native_acceptance_required=True, auto_launch=False)
    validate(p)
    return p


def validate(p):
    validate_base(p)
    e = p['execution']
    require(e['schema'] == 'digit-cache-controller-v3' and e['backend'] == 'pa_sage_cache_policy_v2', 'Wrong controller/backend')
    require(e['physical_gpu'] == GPU and e['gpu_uuid'] == GPU_UUID, 'Changed selected GPU')
    require(Pathlike(e['large_output_root']) == str(LARGE), 'Large outputs must stay in the new /mnt/n0 directory')
    require(e['formal_updates'] == 1179 and e['smoke_updates'] == 4 and e['window_updates'] == 100,
            'Wrong experiment extent')
    require(e['monitor'] == 'nvidia-smi' and e['monitor_max_gap_enforced'] is False, 'Changed monitor policy')
    require(e['monitor_query_timeout_seconds']==5 and e['monitor_period_after_query_seconds']==.5,
            'Monitor timing differs from its implementation')
    require(p['profile']['seed']==23 and p['profile']['batches']==100,'Changed independent presampling extent')
    require(e['all_smokes_before_any_full'] and e['fresh_process_per_arm'] and not e['raw_ssd_writes'],
            'Changed execution isolation/acceptance policy')
    require(e['backend_source_sha256']==sha(ROOT/'candidates/pa_sage_cache_policy_v2/manifest.json'), 'Changed backend source')
    ref=read(p['reference_protocol'])
    require(p['optimizer']==ref['optimizer'] and p['seed']==ref['seed']==0, 'Changed main-experiment model settings')
    for key in ('fanouts','batch_size','hidden','classes','layers','dropout','graph','data','metadata_mode','orders_file'):
        require(p[key]==ref[key], 'Changed fixed workload: '+key)
    require(p['layout']['base']==ref['base_layout'] and p['layout']['overlay']==ref['overlay'], 'Changed fixed layout')
    from candidates.pa_sage_cache_policy_v1.protocol import arms
    require(p['arms']==arms(ref['cpu_cache_rows'],ref['gpu_cache_bytes']), 'Changed approved CPU/GPU capacities')
    require(e['training_examples']==read(ROOT/'configs/papers/protocol.json')['dataset']['splits']['train']['count'],
            'Changed complete-epoch sample count')
    return p


def Pathlike(value):
    from pathlib import Path
    return str(Path(value).resolve())


def schedule(p):
    validate(p)
    return ([dict(stage=x, arm=None) for x in ('bind', 'build', 'rank', 'profile', 'prepare')] +
            [dict(stage='smoke', arm=a) for a in ARMS] +
            [dict(stage='full', arm=a) for a in ARMS] + [dict(stage='aggregate', arm=None)])
