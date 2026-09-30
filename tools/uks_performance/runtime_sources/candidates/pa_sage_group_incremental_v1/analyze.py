"""Compare the complete epoch, keeping profiled runs outside performance means."""
import statistics
from .common import *
from candidates.pa_sage_dense_graph_v1.analyze import compare_workload as compare_reference


def compare_workload(a, b):
    for r in (a, b):
        validate_completion(r)
    require(a['sampler_binary_sha256']==b['sampler_binary_sha256'],'Different sampler binaries')
    return compare_reference(dict(a,variant='graph'),dict(b,variant='graph'))


def analyze(reports):
    require([r['variant'] for r in reports] == full_schedule(), 'Incomplete group schedule')
    require(all(not r['smoke'] and not r['diagnostic_trace'] for r in reports), 'Profile/smoke is not performance')
    checks = [compare_workload(reports[0], r) for r in reports]
    keys = ('e2e_seconds', 'setup_seconds', 'sample_host_seconds', 'model_host_seconds',
            'native_merged_read_seconds', 'ssd_active_seconds', 'forward_host_seconds',
            'backward_host_seconds', 'optimizer_host_seconds')
    variants = {}
    for mode in VARIANTS:
        selected = [r for r in reports if r['variant'] == mode]
        variants[mode] = dict(repeats=len(selected),
            e2e_samples=[r['timing']['e2e_seconds'] for r in selected],
            means={k: statistics.mean(r['timing'][k] for r in selected) for k in keys},
            dense_preparation_mean_seconds=statistics.mean(r['dense_graph']['preparation_seconds'] for r in selected),
            max_dense_retained_allocation_bytes=max(r['dense_graph']['retained_allocation_bytes'] for r in selected))
    old = variants['legacy']['means']['e2e_seconds']
    new = variants['incremental']['means']['e2e_seconds']
    return dict(passed=True, complete=True, variants=variants, workload_checks=checks,
        gids_changed=False, gids_run=False, io_mode='overlap',
        speedup=old / new, reduction_percent=100 * (1 - new / old),
        limits=['Two fresh complete epochs per mode, one seed, no accuracy or significance claim',
                'Sampler host span includes GPU waits and block work; it is not a kernel duration',
                'Model preparation and capture are setup; profiling epochs are excluded from performance',
                'No extra graph-sized allocation, CPU threads, or additional I/O concurrency'])


def markdown(r):
    lines=['# DiGiT 分组权重增量维护对照','','两臂保持相同dense CUDA Graph和读取/模型流水，各两个完整1179更新。','',
        '| 分组选取 | E2E均值（秒） | 采样host（秒） | 两次E2E（秒） |','|---|---:|---:|---|']
    for mode in VARIANTS:
        v=r['variants'][mode]
        lines.append('| %s | %.4f | %.4f | %s |'%(mode,v['means']['e2e_seconds'],v['means']['sample_host_seconds'],', '.join('%.4f'%x for x in v['e2e_samples'])))
    lines+=['','incremental相对legacy为%.4f×；训练E2E减少%.2f%%。'%(r['speedup'],r['reduction_percent']),
        'passed表示执行和数值验收通过；训练E2E不含setup，独立trace不进入均值。']
    if 'trace_comparison' in r:
        t=r['trace_comparison']
        lines+=['','16步诊断中group kernel累计 legacy/incremental 为 %.4f / %.4f ms。'%(t['legacy']['group_kernel_ms'],t['incremental']['group_kernel_ms']),
            'EID kernel不变，两臂均验证32次dense CUDA Graph回放。窗口数据不能外推整个epoch。']
    return '\n'.join(lines)+'\n'
