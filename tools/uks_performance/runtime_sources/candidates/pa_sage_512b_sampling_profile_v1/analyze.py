"""Reconcile nested host spans and report instrumentation overhead separately."""
import statistics
from .common import *


def breakdown(report):
    spans = report['sampling_profile']['spans']
    buckets = dict(dgl_neighbors=0., native_call=0., valid_nonzero=0.,
        compact_indices=0., frontier_graph=0., edge_annotations=0.,
        metadata_lookup=0., output_allocation=0., block_construction=0.,
        storage_annotation=0., sampler_other=0.)
    for path, value in spans.items():
        if not (path == 'sampler' or path.startswith('sampler/')):
            continue
        leaf = path.split('/')[-1]
        if '/storage_annotation' in path:
            category = 'storage_annotation'
        elif leaf.endswith('.dgl_neighbors'):
            category = 'dgl_neighbors'
        elif leaf.endswith('.to_block') or leaf == 'block_build':
            category = 'block_construction'
        elif leaf in buckets:
            category = leaf
        else:
            category = 'sampler_other'
        buckets[category] += value['exclusive_seconds']
    envelope = spans['sampler']['inclusive_seconds']
    require(math.isclose(sum(buckets.values()), envelope, abs_tol=1e-8), 'Nested spans do not reconcile')
    buckets['collate_residual'] = report['timing']['sample_host_seconds'] - envelope
    require(buckets['collate_residual'] >= 0, 'Negative collate residual')
    return buckets


def analyze(reports):
    require([(r['arm'],r['sampling_profile']['mode']) for r in reports]==full_schedule(), 'Incomplete ABBA schedule')
    for r in reports:
        validate_completion(r)
        require(not r['smoke'], 'Smoke timing is not full performance')
    arms = {}
    for arm in ('gids','digit'):
        current = [r for r in reports if r['arm']==arm]
        first = current[0]
        for r in current[1:]:
            for key in ('root_order_sha256','initial_model_sha256','candidate_sha256',
                        'input_binding_sha256','binary_sha256','logical_feature_requests','shapes'):
                require(r[key]==first[key], 'Off/host workload differs: '+arm+'/'+key)
        off = [r for r in current if r['sampling_profile']['mode']=='off']
        host = [r for r in current if r['sampling_profile']['mode']=='host']
        keys = ('e2e_seconds','sample_host_seconds','model_host_seconds','native_merged_read_seconds')
        timings = {mode:{key:statistics.mean(r['timing'][key] for r in group) for key in keys}
                   for mode,group in [('off',off),('host',host)]}
        overhead = {key:100*(timings['host'][key]/timings['off'][key]-1) for key in keys}
        parts = [breakdown(r) for r in host]
        mean_parts = {key:statistics.mean(p[key] for p in parts) for key in parts[0]}
        arms[arm] = dict(timing_means=timings, host_vs_off_percent=overhead,
            off_e2e_samples=[r['timing']['e2e_seconds'] for r in off],
            host_e2e_samples=[r['timing']['e2e_seconds'] for r in host],
            sampling_host_breakdown_seconds=mean_parts,
            loader_row_resolution_seconds=statistics.mean(sum(v['inclusive_seconds']
                for k,v in r['sampling_profile']['spans'].items() if k.endswith('/feature_row_resolution')) for r in host),
            instrumentation_or_run_variation_over_5_percent=abs(overhead['e2e_seconds'])>5)
    for r in reports:
        require(r['root_order_sha256']==reports[0]['root_order_sha256'] and
                r['initial_model_sha256']==reports[0]['initial_model_sha256'], 'Systems differ in roots/model')
    delta = {key:arms['digit']['sampling_host_breakdown_seconds'][key]-arms['gids']['sampling_host_breakdown_seconds'][key]
             for key in arms['gids']['sampling_host_breakdown_seconds']}
    require(math.isclose(sum(delta.values()),
        arms['digit']['timing_means']['host']['sample_host_seconds']-
        arms['gids']['timing_means']['host']['sample_host_seconds'],abs_tol=1e-8), 'Delta accounting mismatch')
    return dict(passed=True, complete=True, arms=arms, digit_minus_gids_host_seconds=delta,
        independent_full_epochs=len(reports), epochs_per_worker=1,
        performance_speedup_off=arms['gids']['timing_means']['off']['e2e_seconds']/arms['digit']['timing_means']['off']['e2e_seconds'],
        limits=['Host scopes locate existing waits, not standalone CUDA kernel time',
                'Native call dispatches selection and original-EID resolution asynchronously',
                'nonzero may include pending CUDA work; do not call it pure compaction cost',
                'Two repetitions per arm/mode do not establish statistical significance',
                'Exclusive host buckets reconcile sampling envelope only; not an E2E waterfall',
                'Profile overhead plus run variation is reported, not subtracted to invent corrected results'])


def markdown(result):
    a=result['arms']; labels={
        'dgl_neighbors':'DGL 邻居采样', 'native_call':'原生分组采样调用（主机提交）',
        'valid_nonzero':'有效输出 nonzero（含可能的等待）', 'compact_indices':'有效节点/目标索引整理',
        'frontier_graph':'DGL frontier 构造', 'edge_annotations':'frontier 边属性整理',
        'metadata_lookup':'已有元数据查找', 'output_allocation':'采样输出分配与参数准备',
        'block_construction':'block 构造', 'storage_annotation':'特征存储地址标注',
        'sampler_other':'采样器其它主机工作', 'collate_residual':'根搬移及 collate 剩余'}
    lines=['# PA/SAGE 512 B 采样细分实测', '', '两臂各 off/host/host/off，每进程 1 个完整 epoch。以下为两次 host 均值，单位秒。', '',
           '| 采样环节 | GIDS | DiGiT | DiGiT − GIDS |','|---|---:|---:|---:|']
    for key,label in labels.items():
        g=a['gids']['sampling_host_breakdown_seconds'][key];d=a['digit']['sampling_host_breakdown_seconds'][key]
        lines.append('| %s | %.6f | %.6f | %+.6f |'%(label,g,d,d-g))
    lines.extend(['', '| 系统 | off E2E | host E2E | host/off 变化 |','|---|---:|---:|---:|'])
    for arm in ('gids','digit'):
        v=a[arm];lines.append('| %s | %.4f | %.4f | %+.2f%% |'%(arm,v['timing_means']['off']['e2e_seconds'],v['timing_means']['host']['e2e_seconds'],v['host_vs_off_percent']['e2e_seconds']))
    lines += ['', '各项只对采样主机段做互斥归账。native_call 是异步提交，nonzero 等位置可能承担前序 GPU 工作的等待；不能据此声称 CUDA 内核耗时。读取阶段的行地址解析另见 JSON，不属于采样段。',
              '', '开启计时相对关闭计时的变化包含插桩开销与运行波动；绝对 E2E 变化超过 5% 时须谨慎解释细分数据，不自动宣称稳定改善。']
    return '\n'.join(lines)+'\n'
