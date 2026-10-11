"""Compare E2E, with async-interface and scheduling controls kept distinct."""
import statistics
from .common import *


def compare_workload(a,b):
    for r in (a,b):validate_completion(r)
    for k in ('smoke','examples','updates','root_order_sha256','initial_model_sha256','candidate_sha256',
              'input_binding_sha256','binary_sha256','logical_feature_requests','shapes','feature_access'):
        require(a[k]==b[k],'Workload differs: '+k)
    # These fields are independent of cache race order and must stay exact.
    for k in ('read_calls','max_merged_batches','max_read_rows','batch_cap_flushes','row_cap_flushes'):
        require(a['pipeline'][k]==b['pipeline'][k],'Accumulator changed: '+k)
    require(len(a['losses'])==len(b['losses']),'Loss count differs')
    delta=max((abs(x-y) for x,y in zip(a['losses'],b['losses'])),default=0.)
    require(all(math.isclose(x,y,rel_tol=1e-5,abs_tol=1e-5) for x,y in zip(a['losses'],b['losses'])),
        'Training losses diverged')
    return dict(passed=True,max_abs_loss_delta=delta,final_model_hash_equal=a['final_model_sha256']==b['final_model_sha256'])


def analyze(reports):
    require([r['variant'] for r in reports]==full_schedule(),'Incomplete mode schedule')
    require(all(not r['smoke'] and not r['diagnostic_trace'] for r in reports),'Profile/smoke is not performance')
    checks=[compare_workload(reports[0],r) for r in reports]
    result={}
    keys=('e2e_seconds','setup_seconds','sample_host_seconds','model_host_seconds','native_merged_read_seconds')
    for mode in VARIANTS:
        selected=[r for r in reports if r['variant']==mode]
        result[mode]=dict(repeats=len(selected),e2e_samples=[r['timing']['e2e_seconds'] for r in selected],
            means={k:statistics.mean(r['timing'][k] for r in selected) for k in keys})
    overlap=result['overlap']['means']['e2e_seconds']
    return dict(passed=True,complete=True,variants=result,workload_checks=checks,gids_changed=False,gids_run=False,
        speedup_vs_legacy=result['legacy']['means']['e2e_seconds']/overlap,
        speedup_vs_async_serial=result['serial']['means']['e2e_seconds']/overlap,
        limits=['Two independent epochs per mode; no significance/accuracy claim',
                'Async read and wait host times are not GPU kernel durations; inspect trace for actual overlap',
                'Sampling remains on the model stream, no extra CPU threads/cores',
                'Bounded one-batch scheduling lookahead; shared cache capacity unchanged'])


def markdown(r):
    lines=['# DiGiT 读取与模型重叠对照','','每模式两次完整 1179 更新；另一次带 CUDA trace 的完整 epoch 不计入性能。','',
        '| 模式 | E2E 均值（秒） | 两次原值（秒） |','|---|---:|---|']
    for mode in VARIANTS:
        v=r['variants'][mode];lines.append('| %s | %.4f | %s |'%(mode,v['means']['e2e_seconds'],', '.join('%.4f'%x for x in v['e2e_samples'])))
    lines+=['','overlap 相对 legacy / async serial 为 %.4f× / %.4f×。'%(r['speedup_vs_legacy'],r['speedup_vs_async_serial']),
        '模型、采样和缓存容量保持相同。GPU 时间线上有重叠不等于完整 E2E 一定下降。']
    if 'overlap_trace' in r:
        t=r['overlap_trace']
        lines+=['','真实读 kernel 与模型 kernel 的重叠已观察到：%s；16 步窗口交叠并集 %.4f ms。'%(
            '是' if t['real_gpu_overlap_observed'] else '否',t['overlap_union_ms']),
            'passed 表示执行与数值验收通过；若未观察到交叠或 E2E 变慢，保留为负结果。']
    return '\n'.join(lines)+'\n'
