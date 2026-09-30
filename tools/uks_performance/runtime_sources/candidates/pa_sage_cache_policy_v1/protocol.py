"""Paper Fig.13-style policies; no implicit native execution or graph rebuilding."""
import copy
from .common import ROOT, REFERENCE, ARMS, LABELS, read, sha, require


def arms(cpu_rows, gpu_bytes):
    require(type(cpu_rows) is int and cpu_rows >= 0, 'Invalid CPU row budget')
    require(type(gpu_bytes) is int and gpu_bytes >= 0 and gpu_bytes % 4096 == 0, 'Invalid GPU byte budget')
    return {name: dict(label=LABELS[name], selection='freq' if name == 'digit' else name,
                      cpu_rows=cpu_rows, cpu_feature_bytes=cpu_rows * 512,
                      gpu_feature_cache_bytes=gpu_bytes if name == 'digit' else 0,
                      gpu_policy='fifo' if name == 'digit' else 'bypass',
                      lookup_order=['gpu', 'cpu', 'ssd'] if name == 'digit' else ['cpu', 'ssd'])
            for name in ARMS}


def compile_protocol():
    ref = read(REFERENCE)
    pool_path = ROOT / 'candidates/pa_sage_layout_shared_resume_v3/pools.json'
    pool = read(pool_path)['real']
    receipt = ROOT / 'results/pa_sage_layout_shared_20260925_v1/pools/real_receipt.json'
    bundle = ROOT / ref['base_layout'] / 'final/bundle/manifest.json'
    overlay = ROOT / ref['overlay'] / 'overlay_receipt.json'
    result = dict(schema='digit-pa-cache-policy-preparation-v1', dataset='PA', model='SAGE',
        paper='atc26-paper1508.pdf sections 4.4 and 5.4.4, Figure 13',
        policy_decision='author_selected_static_topk_vs_frequency_plus_fifo',
        reference_protocol=str(REFERENCE), reference_protocol_sha256=sha(REFERENCE),
        native_candidate_sha256=sha(ROOT / 'candidates/pa_sage_layout_shared_resume_v3/manifest.json'),
        epochs=1, evaluation='disabled', repeats=1, warmup_batches=0,
        arms=arms(ref['cpu_cache_rows'], ref['gpu_cache_bytes']), run_order=list(ARMS),
        profile=dict(batches=100, seed=23, counter='one count per unique logical input node per batch',
                     source='independent presampling on the same fixed graph/layout/fanouts',
                     exclusion='no measured worker, validation or test used to select the hot set',
                     rng_reset='fresh native process per arm; seed0 model and measurement sampler',
                     frequencies_shared_between=['freq', 'digit']),
        ranking=dict(ties='descending score then ascending logical node ID', unit='logical feature row',
                     degree='incoming degree on the fixed bidirectional training CSC, including multiedges and self loops',
                     revpr=dict(graph='reverse of the same training CSC', damping=.85, iterations=20,
                                dangling='uniform redistribution', dtype='float64'),
                     graph_note='Bidirectional PA CSC is symmetric; reverse PageRank is PageRank on that graph. Do not substitute an archived score vector from a different directed graph.'),
        layout=dict(base=ref['base_layout'], overlay=ref['overlay'], point=copy.deepcopy(ref['point']),
                    manifest_sha256=sha(bundle), overlay_receipt_sha256=sha(overlay),
                    policy='fixed for all arms, including the historical layout hot exclusions; no per-policy regrouping'),
        features=dict(mode='logical_node_real', row_bytes=512, pool_offset=pool['offset'],
                      verified_bytes=pool['verify_bytes'], pool_spec_sha256=sha(pool_path),
                      receipt_path=str(receipt), receipt_sha256=sha(receipt),
                      replicas='all copies of a logical hot node map to its single CPU feature slot',
                      padding='uncached; selecting an individual row must not cache its cold page neighbours',
                      raw_ssd_writes=False, new_feature_payload=False),
        metrics=dict(denominator='logical feature row requests returned by sampling, after within-batch deduplication',
                     partition='cpu_served_rows + gpu_hit_rows + ssd_served_rows == logical_requests',
                     combined_hit='(cpu_served_rows + gpu_hit_rows) / logical_requests',
                     cpu_hit='cpu_served_rows / logical_requests', gpu_hit='gpu_hit_rows / logical_requests',
                     conditional_gpu_hit='gpu_hit_rows / gpu_ssd_route_rows, reported separately; not the overall hit ratio',
                     io='SSD primary/completed/replay commands and bytes; distinct from logical SSD-served rows'),
        measurement=dict(kind='complete first epoch in a fresh process',
                         separate_costs=['degree/RevPR', 'presampling', 'TopK', 'CPU preload', 'model/graph setup', 'root order'],
                         one_epoch_expected_updates=1179, precision_or_convergence_claim=False),
        capacity_note='Same CPU feature capacity in all arms. DiGiT additionally has 4 GiB GPU feature cache. This compares the paper policy packages, not equal-total-capacity algorithms. Metadata and staging are separate.',
        implementation=dict(native_ready=False, auto_launch=False, existing_backend_supports_static_bypass=False,
                            blockers=['static-only direct-I/O bypass with native route counters',
                                      'exact logical-row CPU cache integrated with current useful-I/O instrumentation',
                                      'fresh full-graph rankings and independent frequency profile',
                                      'isolated native build, budget checks and short acceptance after the active grid']))
    for key in ('seed', 'fanouts', 'batch_size', 'optimizer', 'hidden', 'classes', 'layers', 'dropout',
                'graph', 'metadata_mode', 'data', 'orders_file', 'bfs_enabled'):
        result[key] = copy.deepcopy(ref[key])
    validate(result)
    return result


def validate(p):
    require(p['schema'] == 'digit-pa-cache-policy-preparation-v1', 'Wrong schema')
    require(p['epochs'] == 1 and p['evaluation'] == 'disabled' and p['warmup_batches'] == 0, 'Wrong performance extent')
    require(p['run_order'] == list(ARMS) and set(p['arms']) == set(ARMS), 'Wrong policy matrix')
    cpu = p['arms']['freq']['cpu_rows']
    gpu = p['arms']['digit']['gpu_feature_cache_bytes']
    require(p['arms'] == arms(cpu, gpu) and gpu > 0, 'Changed paper policy semantics/budget')
    require(p['features']['mode'] == 'logical_node_real' and not p['features']['raw_ssd_writes'],
            'Logical alias caching requires equal replica values; proxy feature rows are incompatible')
    require(p['profile']['seed'] != p['seed'] and 0 < p['profile']['batches'] < 1179, 'Wrong independent profile')
    require(p['profile']['frequencies_shared_between'] == ['freq', 'digit'], 'Freq and DiGiT need the same ranking')
    require(p['features']['row_bytes'] == 512 and p['batch_size'] == 1024 and p['fanouts'] == [10, 5, 5], 'Changed training configuration')
    require(p['layout']['point']['id'] == 'g2_r20', 'Expected fixed g2/r20 layout')
    return p
