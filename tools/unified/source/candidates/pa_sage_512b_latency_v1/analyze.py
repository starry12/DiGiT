"""Matched workload checks and four-variant E2E comparison."""
import statistics
from .common import *


def compare_workload(reference,other):
    for r in (reference,other):validate_completion(r)
    for key in ('smoke','examples','updates','root_order_sha256','initial_model_sha256',
                'candidate_sha256','input_binding_sha256','binary_sha256','logical_feature_requests','shapes'):
        require(reference[key]==other[key],'Workload differs: '+key)
    require(len(reference['losses'])==len(other['losses']),'Loss sequence length differs')
    deltas=[abs(a-b) for a,b in zip(reference['losses'],other['losses'])]
    close=all(math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-5) for a,b in zip(reference['losses'],other['losses']))
    require(close,'Loss sequence diverged beyond frozen tolerance')
    return dict(passed=True,max_abs_loss_delta=max(deltas,default=0.),losses_allclose=close,
        final_model_hash_equal=reference['final_model_sha256']==other['final_model_sha256'])


def analyze(reports):
    require([r['variant'] for r in reports]==full_schedule(),'Incomplete four-variant schedule')
    require(all(not r['smoke'] for r in reports),'Cannot report smoke as performance')
    parity=[compare_workload(reports[0],r) for r in reports]
    metrics=('e2e_seconds','sample_host_seconds','model_host_seconds','forward_host_seconds',
             'backward_host_seconds','optimizer_host_seconds','loss_zero_grad_host_seconds',
             'native_merged_read_seconds','block_conversion_host_seconds','setup_seconds')
    variants={}
    for name in VARIANTS:
        selected=[r for r in reports if r['variant']==name]
        variants[name]=dict(repeats=len(selected),e2e_samples=[r['timing']['e2e_seconds'] for r in selected],
            means={k:statistics.mean(r['timing'][k] for r in selected) for k in metrics})
    base=variants['baseline']['means']['e2e_seconds']
    for v in variants.values():
        v['speedup_vs_original_digit']=base/v['means']['e2e_seconds']
        v['e2e_reduction_percent']=100*(1-v['means']['e2e_seconds']/base)
    return dict(passed=True,complete=True,variants=variants,workload_checks=parity,
        epochs_per_process=1,updates_per_process=1179,gids_changed=False,gids_run=False,
        limits=['Two runs per variant, single seed, no statistical-significance or accuracy claim',
                'Unchanged GIDS results remain a separate historical baseline',
                'Compare full E2E, not sums of asynchronous/nested host spans',
                'No new CUDA synchronization; setup remains separate'])


def markdown(result):
    names=dict(baseline='原 DiGiT',frontier='仅 frontier',functional='仅 functional SAGE',combined='组合')
    lines=['# DiGiT frontier / functional mean-SAGE 验证结果','','每项两个独立完整 epoch，均为 1179 更新。GIDS 实现和已验收结果不变。','',
           '| 实现 | E2E（秒） | 采样主机段 | 模型主机段 | 相对原 DiGiT |','|---|---:|---:|---:|---:|']
    for name in VARIANTS:
        r=result['variants'][name];m=r['means']
        lines.append('| %s | %.4f | %.4f | %.4f | %.4f× |'%(names[name],m['e2e_seconds'],m['sample_host_seconds'],m['model_host_seconds'],r['speedup_vs_original_digit']))
    lines += ['', '两次重复的原始 E2E、各段数据及 loss/工作量核对见 summary.json。分段计时可能异步或嵌套，不能直接相加解释 E2E；单项或组合若变慢也照实保留。']
    return '\n'.join(lines)+'\n'
