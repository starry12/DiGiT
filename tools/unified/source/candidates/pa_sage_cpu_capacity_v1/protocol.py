"""Figure 14 first slice: PA/SAGE, CPU 0/5/10/20%, constant GPU FIFO."""
from candidates.pa_sage_cache_policy_v1.protocol import compile_protocol
from .common import ROOT, LARGE, GPU, GPU_UUID, ARMS, PERCENTAGES, read, require, native_source_id


def arms(nodes, gpu_bytes=4*2**30):
    require(type(nodes) is int and nodes > 0, 'Invalid logical feature population')
    require(gpu_bytes == 4*2**30, 'GPU feature cache must remain 4 GiB')
    return {name: dict(label='DiGiT CPU %d%%' % percent, cpu_percent=percent,
        selection='freq', cpu_rows=nodes*percent//100,
        cpu_feature_bytes=(nodes*percent//100)*512, gpu_feature_cache_bytes=gpu_bytes,
        gpu_policy='fifo', lookup_order=['gpu', 'cpu', 'ssd'])
        for name, percent in zip(ARMS, PERCENTAGES)}


def compile_plan():
    p = compile_protocol()
    p.update(schema='digit-pa-cpu-capacity-v1',
        paper='atc26-paper1508.pdf sections 4.4 and 5.4.5, Figure 14',
        policy_decision='vary only exact CPU feature capacity; shared independent Freq ranking; fixed GPU FIFO',
        arms=arms(p['graph']['nodes']), run_order=list(ARMS),
        ranking=dict(ties='descending frequency then ascending logical node ID', unit='logical feature row', nested=True),
        capacity_note='Denominator: original logical nodes * 512 bytes, excluding replicas, padding and metadata. '
            'K=floor(N*percent/100), no page expansion. Same 4 GiB GPU FIFO and fixed graph/layout at every point. '
            'Zero CPU rows allocate no CPU feature backing and perform no preload I/O; row-map metadata remains fixed.',
        scope=dict(implemented=['PA/SAGE'], deferred_datasets=['UKS', 'UKL', 'CL', 'IG'], paper_matrix_complete=False))
    p['profile']['frequencies_shared_between'] = list(ARMS)
    p['measurement']['separate_costs'] = ['presampling/count flush', 'TopK/nesting/seal', 'CPU cache allocation/preload/map installation', 'model/graph setup', 'root order']
    p['execution'] = dict(schema='digit-cpu-capacity-controller-v1', backend='pa_sage_cpu_capacity_v1',
        backend_abi=2, backend_source_sha256=native_source_id(),
        large_output_root=str(LARGE), physical_gpu=GPU, gpu_uuid=GPU_UUID,
        monitor='nvidia-smi', monitor_query_timeout_seconds=5, monitor_period_after_query_seconds=.5,
        monitor_max_gap_enforced=False, smoke_updates=4, window_updates=100,
        formal_updates=1179, training_examples=read(ROOT/'configs/papers/protocol.json')['dataset']['splits']['train']['count'],
        stages=['bind', 'build', 'profile', 'prepare', 'smoke', 'full', 'aggregate'],
        all_smokes_before_any_full=True, fresh_process_per_arm=True,
        full_trace='two position-weighted int64 GPU reductions per sampled tensor; CPU SHA256 after epoch',
        trace_overhead='included equally in each measured epoch, not an exact cryptographic proof of every GPU tensor',
        raw_ssd_writes=False, automatic_start=False)
    p['implementation'] = dict(controller_source_ready=True, native_ready=False, native_acceptance_required=True, auto_launch=False)
    return p


def validate(p):
    # No arbitrary knob overrides in this first slice; a changed workload needs a new plan.
    require(p == compile_plan(), 'CPU capacity protocol differs from the fixed PA/SAGE plan')
    return p


def schedule(p):
    validate(p)
    return ([dict(stage=x, arm=None) for x in ('bind', 'build', 'profile', 'prepare')] +
            [dict(stage='smoke', arm=a) for a in ARMS] +
            [dict(stage='full', arm=a) for a in ARMS] + [dict(stage='aggregate', arm=None)])
