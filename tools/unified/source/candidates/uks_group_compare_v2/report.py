"""All five same-round ratios, with their median as the primary result."""
import math
import statistics
from .common import require, ROUNDS, VARIANTS


def validate_worker(r, mode, variant, source, prepared, number):
    require(r.get('passed') and r['mode'] == mode and r['matched_variant'] == variant, 'Wrong worker')
    require(r['candidate_sha256'] == source and r['prepared_sha256'] == prepared, 'Wrong source/input')
    require(r['round'] == (number or 0), 'Wrong round')
    require(r['bfs_enabled'] and r['order_policy'] == 'full_training_induced_bfs_block_shuffle_window', 'Wrong root policy')
    require(r['anonymous_host_csc'] and not r['raw_ssd_writes'] and not r['feature_layout_changed'], 'Unsafe input route')
    require(r['performance_result'] == (mode == 'performance'), 'Wrong timing classification')
    if mode == 'smoke':
        require(r['feature_bit_exact'] and r['sample_edges_verified'], 'Missing correctness evidence')
    else:
        require(r['updates'] == 320 and r['warmup_batches'] == 20 and r['measured_batches'] == 300, 'Incomplete window')
        require(r['roots_sha256'] == r['matched_window_roots_sha256'], 'Unmatched root sequence')
        require(math.isfinite(r['seconds']) and r['seconds'] > 0 and r['training']['reconciled'], 'Invalid timing/counters')
    if mode != 'smoke':
        require(r['hot_policy'] == ('revpr' if variant == 'gids' else 'freq100_seed23'),
                'Changed CPU hot-set policy')
    region = r['region'] if mode == 'smoke' else r['training']
    if variant == 'gids':
        require(region['request_sizes']['primary_2k'] == 0, 'GIDS unexpectedly uses grouped reads')


def summarize(reports):
    baseline = reports['round1_gids']
    rows = []
    for number in range(1, ROUNDS+1):
        arms = {}
        for variant in VARIANTS:
            r = reports['round%d_%s' % (number, variant)]
            for name in ('roots_sha256', 'matched_window_roots_sha256', 'initial_model_sha256', 'order_policy'):
                require(r[name] == baseline[name], 'Unmatched '+name)
            for name in ('cpu_rows', 'cpu_feature_bytes', 'gpu_feature_cache_bytes'):
                require(r['cache'][name] == baseline['cache'][name], 'Unmatched cache capacity: '+name)
            reference = reports['round1_'+variant]
            require(r['hot_file_sha256'] == reference['hot_file_sha256'] and
                    r['hot_policy'] == reference['hot_policy'] and r['cache'] == reference['cache'],
                    'Cache policy/input changed between rounds')
            region = r['training']; sizes = region['request_sizes']
            one, two = sizes['primary_1k'], sizes['primary_2k']
            arms[variant] = dict(seconds=r['seconds'], measured_batches=r['measured_batches'],
                feature_rows=region['serving']['logical_requests'], serving=region['serving'],
                primary_1k=one, primary_2k=two, ssd=region['device'],
                paired_command_fraction=two/(one+two) if one+two else 0,
                grouped_primary_row_fraction=2*two/(one+2*two) if one+two else 0,
                hot_policy=r['hot_policy'], cache=r['cache'], affinity=r['affinity'])
        old, reuse = (reports['round%d_%s' % (number, v)] for v in ('v1', 'v2'))
        require(old['hot_file_sha256'] == reuse['hot_file_sha256'] and old['cache'] == reuse['cache'],
                'DiGiT variants differ in hot set or capacity')
        rows.append(dict(round=number, arms=arms,
            speedup_v1_over_v2=arms['v1']['seconds']/arms['v2']['seconds'],
            speedup_gids_over_v1=arms['gids']['seconds']/arms['v1']['seconds'],
            speedup_gids_over_v2=arms['gids']['seconds']/arms['v2']['seconds']))
    ratios = {v: [r['speedup_gids_over_'+v] for r in rows] for v in ('v1', 'v2')}
    return dict(passed=True, complete=True, rounds=rows, paired_round_speedups=ratios,
        aggregation='median_of_five_paired_round_speedups',
        median_speedup={v: statistics.median(values) for v, values in ratios.items()},
        median_speedup_v1_over_v2=statistics.median(r['speedup_v1_over_v2'] for r in rows),
        median_seconds={v: statistics.median(r['arms'][v]['seconds'] for r in rows) for v in VARIANTS},
        paired_roots=True, paired_initial_model=True, paired_cache_capacity=True,
        roots_sha256=baseline['roots_sha256'],
        limits=['Five fresh-process rounds, seed 0, identical BFS warmup and timed roots; not five seeds',
                'Matched output examples, not identical sampled neighbors: sampling algorithms differ',
                'GIDS uses RevPR; both DiGiT variants use Freq; equal feature cache capacity',
                'Inherited CPU affinity: GIDS original mask, DiGiT CPU2; recorded per arm',
                'Other-GPU non-DiGiT jobs allowed by user; host CPU and I/O isolation not claimed',
                'UKS synthetic features/labels; no real-data accuracy claim',
                'Setup, smoke and preparation excluded from timed 300-batch windows'])


def coverage_summary(receipts):
    a,b=(receipts[name]['validation'] for name in ('v1','v2'))
    require(a['passed'] and b['passed'],'Unvalidated adjacency')
    for key in ('nodes','edges','groups','old_group_references'):
        require(a[key]==b[key],'Different coverage graph: '+key)
    require(b['v1_greedy_added_references']==a['added_group_references'],
            'v2 greedy reconstruction differs from accepted v1')
    require(b['added_group_references']>=a['added_group_references'],'Coverage regressed')
    x,y=(v['new_group_covered_edges']/v['edges'] for v in (a,b))
    return dict(v1_fraction=x,v2_fraction=y,delta_percentage_points=100*(y-x),
                additional_group_references=b['added_group_references']-a['added_group_references'],
                owners_improved_over_v1=b['owners_improved_over_v1'],
                physical_groups=b['groups'],group_size=2,feature_layout_changed=False)
