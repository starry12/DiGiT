"""Compare the complete epoch, keeping profiled runs outside performance means."""
import statistics
from .common import *
from candidates.pa_sage_512b_overlap_v1.analyze import compare_workload as compare_reference


def compare_workload(a, b):
    for r in (a, b):
        validate_completion(r)
    return compare_reference(dict(a, variant='overlap',model_implementation='functional_mean_sage'), dict(b, variant='overlap',model_implementation='functional_mean_sage'))


def analyze(reports):
    require([r['variant'] for r in reports] == full_schedule(), 'Incomplete dense schedule')
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
    old = variants['eager']['means']['e2e_seconds']
    new = variants['graph']['means']['e2e_seconds']
    return dict(passed=True, complete=True, variants=variants, workload_checks=checks,
        gids_changed=False, gids_run=False, io_mode='overlap',
        speedup=old / new, reduction_percent=100 * (1 - new / old),
        limits=['Two fresh complete epochs per mode, one seed, no accuracy or significance claim',
                'Sampler host span includes GPU waits and block work; it is not a kernel duration',
                'Model preparation and capture are setup; profiling epochs are excluded from performance',
                'No extra graph-sized allocation, CPU threads, or additional I/O concurrency'])


def markdown(r):
    lines = ['# DiGiT 末层 dense CUDA Graph 对照', '',
        '两臂均使用原解析与读取/模型流水，仅替换末层 dense 前后向；各两次完整1179更新。', '',
        '| 模式 | E2E均值（秒） | 模型host（秒） | 两次E2E（秒） |',
        '|---|---:|---:|---|']
    for mode in VARIANTS:
        v = r['variants'][mode]
        lines.append('| %s | %.4f | %.4f | %s |' % (mode, v['means']['e2e_seconds'],
            v['means']['model_host_seconds'], ', '.join('%.4f' % x for x in v['e2e_samples'])))
    lines += ['', 'graph相对eager为%.4f×，训练E2E减少%.2f%%。' % (r['speedup'], r['reduction_percent']),
        'passed表示执行与数值验收；训练E2E含静态缓冲拷贝、不含捕获准备。任何回退保留。']
    if 'trace_comparison' in r:
        t = r['trace_comparison']
        lines += ['', '16步独立诊断窗口：eager/graph CUDA Graph launch数为 %d/%d。' % (
            t['eager']['graph_launches'], t['graph']['graph_launches']),
            '诊断窗口排除在性能均值之外，graph launch出现只证明实际使用捕获回放。']
    return '\n'.join(lines) + '\n'
