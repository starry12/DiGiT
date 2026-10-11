"""PA-only memory accounting, bound to the audited layout and deployed ABI."""
import math
from ae.pa_sage.common import ROOT, PROTOCOL, cfg, source_config, read, sha, require, setup_imports

PROFILE = ROOT / 'configs/paper/pa_sage_admission_v4.json'


def estimate_bound(manifest, protocol, source, profile):
    """Pure calculation; callers must verify profile identities before use."""
    setup_imports()
    from digit.gpu_admission import estimate
    p, c = protocol['paper_specified'], protocol['reconstruction_choices']
    require(c['gpu_cache_bytes'] == 4*2**30 and c['io_page_bytes'] == 4096,
            'Audited profile requires the unchanged 4 GiB cache and 4 KiB pages')
    require(profile['data_page_bytes'] == 32 and profile['labels_reserve_bytes'] == 64*2**20,
            'Unsupported audited ABI or label reserve')
    plan = estimate(manifest, batch_size=p['batch_size'], fanouts=p['fanouts'],
                    hidden=p['hidden_dim'], classes=c['num_classes'], cache_mib=4096,
                    reserve_gib=4, metadata_mode='gpu_i32')
    # Retain PA preparation's host allowances for CPU preload and node arrays.
    plan['host_required_bytes'] += 2*c['cpu_cache_rows']*512 + 2*manifest['dataset']['num_nodes']*8
    parts = plan['components_bytes']
    parts['common_degree_counter'] = manifest['dataset']['num_nodes']*4
    old_required = math.ceil(sum(parts.values())*plan['safety_multiplier'])
    # Two old/new copies of each epoch's roots and labels, with evaluation
    # output overlap and 1 MiB slack. Full labels/masks remain on the host.
    counts = {k: source['splits'][k]['count'] for k in ('train', 'valid', 'test')}
    overlap = 4*counts['train']*8 + 4*max(counts['valid'], counts['test'])*8 + 2**20
    require(overlap <= profile['labels_reserve_bytes'], 'Label overlap exceeds audited reserve')
    payload = manifest['feature']['num_storage_rows']*manifest['feature']['dim']*4
    require(payload % c['io_page_bytes'] == 0, 'Unaligned storage geometry')
    parts['bam_storage_range_allowance'] = (payload//c['io_page_bytes'])*profile['data_page_bytes']
    parts['labels_masks_seed_indices'] = profile['labels_reserve_bytes']
    required = math.ceil(sum(parts.values())*plan['safety_multiplier'])
    require(required == profile['expected_required_bytes'], 'Audited geometry/budget changed')
    require(plan['host_required_bytes'] == profile['base_host_required_bytes'], 'Base host geometry changed')
    require(profile['trace_and_monitor_host_reserve_bytes'] == 2**30, 'Wrong trace/monitor host reserve')
    plan['base_host_required_bytes'] = plan['host_required_bytes']
    plan['host_required_bytes'] += profile['trace_and_monitor_host_reserve_bytes']
    plan['trace_and_monitor_host_reserve_bytes'] = profile['trace_and_monitor_host_reserve_bytes']
    require(plan['host_required_bytes'] == profile['expected_host_required_bytes'], 'Host geometry changed')
    plan.update(schema='digit-pa-sage-admission-v4', required_bytes=required,
                historical_required_bytes=old_required, label_overlap_bytes=overlap,
                scope='native smoke admission; full training additionally requires paired native receipts',
                audit_summary_sha256=profile['audit_summary_sha256'])
    return plan


def audited_plan(data):
    profile = read(PROFILE)
    require(profile['schema'] == 'digit-pa-sage-admission-profile-v4', 'Wrong admission profile')
    require(sha(PROTOCOL) == profile['protocol_sha256'], 'Admission protocol changed')
    manifest = data/'final/bundle/manifest.json'
    require(sha(manifest) == profile['manifest_sha256'], 'Admission layout changed')
    for name, expected in profile['implementation_sha256'].items():
        require(sha(ROOT/name) == expected, 'Audited implementation changed: '+name)
    plan = estimate_bound(read(manifest), cfg(), source_config(), profile)
    plan['profile_sha256'] = sha(PROFILE)
    return plan
