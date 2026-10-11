"""Compare the complete epoch, keeping profiled runs outside performance means."""
import statistics
from .common import *
from candidates.pa_sage_group_incremental_v1.analyze import compare_workload as compare_reference


def compare_workload(a, b):
    for r in (a, b):
        validate_completion(r)
    require(a['sampler_binary_sha256']==b['sampler_binary_sha256'],'Different sampler binaries')
    require(a['normalization_binary_sha256']==b['normalization_binary_sha256'],'Different normalization binaries')
    result=compare_reference(as_reference(a),as_reference(b))
    require(result['max_abs_loss_delta']==0 and result['final_model_hash_equal'],'Mean normalization changed training bits')
    return result


def analyze(reports):
    require([r['variant'] for r in reports] == full_schedule(), 'Incomplete mean schedule')
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
    new = variants['fused']['means']['e2e_seconds']
    return dict(passed=True, complete=True, variants=variants, workload_checks=checks,
        gids_changed=False, gids_run=False, io_mode='overlap',
        speedup=old / new, reduction_percent=100 * (1 - new / old),
        limits=['Two fresh complete epochs per mode, one seed, no accuracy or significance claim',
                'Sampler host span includes GPU waits and block work; it is not a kernel duration',
                'Model preparation and capture are setup; profiling epochs are excluded from performance',
                'No extra graph-sized allocation, CPU threads, or additional I/O concurrency'])


def markdown(r):
    lines=['# DiGiT mean归一化融合对照','','两臂保持增量分组选取、原EID、dense Graph和读取/模型流水；各两次完整1179更新。','',
        '| 模式 | E2E（秒） | 采样host（秒） | 模型host（秒） | 两次E2E（秒） |','|---|---:|---:|---:|---|']
    for mode in VARIANTS:
        v=r['variants'][mode];m=v['means']
        lines.append('| %s | %.4f | %.4f | %.4f | %s |'%(mode,m['e2e_seconds'],m['sample_host_seconds'],m['model_host_seconds'],', '.join('%.4f'%x for x in v['e2e_samples'])))
    lines+=['','fused相对legacy为%.4f倍；训练E2E减少%.2f%%。'%(r['speedup'],r['reduction_percent']),
        'passed表示执行及逐位数值验收通过；训练计时不含setup，独立trace不进入性能均值。']
    if 'trace_comparison' in r:
        t=r['trace_comparison']
        lines+=['','16步窗口fused normalization forward/backward调用为%d/%d；两臂均有32次dense Graph launch。'%(t['fused']['mean_forward_calls'],t['fused']['mean_backward_calls']),
            '窗口只用于确认实际执行路径及定位，不能外推完整epoch。']
    return '\n'.join(lines)+'\n'
