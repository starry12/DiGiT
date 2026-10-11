"""Accept paired real-device smoke evidence before enabling full PA training."""
import math
from pathlib import Path
from ae.pa_sage.common import ROOT, PROTOCOL, cfg, read, sha, require
from ae.pa_sage.admission import PROFILE


def current_binding(data):
    from ae.common import check_payload
    baseline = check_payload('papers_gids')
    full = read(data/'ssd_ready.json')
    require(sha(full['state']) == full['state_sha256'], 'SSD state changed')
    return dict(execution_profile_sha256=sha(ROOT/'configs/paper/pa_sage_execution_v4.json'),execution_sha256=sha(ROOT/'ae/release_manifest.json'),
                protocol_sha256=sha(PROTOCOL), prepared_sha256=sha(data/'prepared.json'),
                admission_profile_sha256=sha(PROFILE),
                ssd_binding=dict(ready_sha256=sha(data/'ssd_ready.json'),
                    baseline_state_sha256=sha(baseline['state']),
                    full_state_sha256=sha(full['state']), full_offset=full['offset']))


def validate_reports(reports, binding):
    """Validate measured reports, never accept only a top-level passed flag."""
    require(set(reports) == {'gids', 'digit_full'}, 'Both native arms are required')
    for arm, report in reports.items():
        require(report['passed'] is True and report['smoke'] is True and report['source_only'] is False,
                'Require successful real-device smoke, not filesystem features or full training')
        require(report['arm'] == arm and report['seed'] == 0 and report['updates'] == 4, 'Wrong smoke extent')
        require(report['raw_writes'] is False and report['bfs_enabled'] is False, 'Wrong I/O/order contract')
        require(report['paired_random_order'] and report['all_epochs_metadata_and_cache_reused'], 'Reuse/order check missing')
        for key, expected in binding.items():
            require(report[key] == expected, 'Stale native report: '+key)
        require(report['cache_bytes'] == 4*2**30 and report['cpu_cache_rows'] == cfg()['reconstruction_choices']['cpu_cache_rows'], 'Cache budget mismatch')
        for end in ('initial_cache', 'final_cache'):
            require(report[end]['gpu']['capacity_pages'] == 1048576, 'Actual GPU cache is not 4 GiB')
        require(report['final_cache']['device']['completed_commands'] > report['initial_cache']['device']['completed_commands'], 'Smoke performed no SSD reads')
        require(report['admission']['passed'] and report['admission']['profile_sha256'] == binding['admission_profile_sha256'], 'Wrong admission report')
        resource = report['resource_observations']
        stages = {x['stage'] for x in resource['checkpoints']}
        require({'start', 'native_cache_allocated', 'cpu_preloaded', 'graph_loaded', 'degrees_loaded', 'training_complete'} <= stages, 'Resource checkpoints missing')
        require(resource['observed_peak_device_used_bytes'] <= report['admission']['required_bytes'], 'Observed usage exceeds audited budget')
        require(resource['mode']=='checkpoints_only_external_sampler' and not resource['background_monitor_in_worker'] and resource['monitor_samples']==0, 'Worker monitoring is not isolated')
        monitor=report['external_monitor']
        require(monitor['sample_count']>0 and monitor['monitor_pid']!=monitor['worker_pid'] and monitor['monitor_peak_rss_bytes']<=64*2**20 and monitor['observed_peak_device_used_bytes']<=report['admission']['required_bytes'], 'External device monitor incomplete or exceeds budget')
        lifecycle=report['evaluation_lifecycle']
        require(lifecycle['validation_calls']==2 and lifecycle['test_calls']==0 and lifecycle['diagnostic_replays']==0 and lifecycle['training_rng_unchanged'], 'Wrong formal smoke evaluation lifecycle')
        require(len(report['epochs']) == 2 and len(report['audits']) == 4 and report['test'] is None, 'Incomplete smoke evidence')
        require(len({(a['epoch'], a['batch']) for a in report['audits']}) == 4, 'Duplicate audit records')
        require({(a['epoch'], a['batch']) for a in report['audits']} == {(e,b) for e in range(2) for b in range(2)}, 'Wrong audit batch IDs')
        for audit in report['audits']:
            require(math.isfinite(audit['loss']) and len(audit['feature']) == 64, 'Bad feature/loss audit')
            if arm == 'digit_full':
                require(len(audit['storage_rows']) == 64 and len(audit['storage_flags']) == 64, 'Missing logical/storage mapping check')
        group_edges = 0
        for index, epoch in enumerate(report['epochs']):
            require(epoch['epoch'] == index+1 and epoch['updates'] == 2, 'Wrong epoch extent')
            require(epoch['metadata_reused'] and epoch['cache_object_reused'], 'Lost persistent state')
            require(len(epoch['losses']) == 2 and all(math.isfinite(x) for x in epoch['losses']), 'Nonfinite training')
            val = epoch['validation']
            require(val['training_rng_restored'] and val['batches'] == 2 and val['examples'] == 2048, 'Incomplete validation/RNG check')
            require(len(val['bindings']) == 2 and math.isfinite(val['loss']), 'Missing validation feature audits')
            require(len(epoch['windows']) == 1, 'Missing two-batch I/O window')
            group_edges += sum(w['group_edges'] for w in epoch['windows'])
            for interval in (epoch['training'], val):
                require(interval['device']['enabled'] and interval['device']['reconciled'], 'Device I/O counters not reconciled')
                require(interval['reconciliation']['reconciled'], 'Mixed I/O reconciliation missing')
                if arm == 'digit_full':
                    require(interval['reconciliation']['enabled'] and interval['reconciliation']['device_counters_checked'], 'Grouped device path not checked')
            require(0 <= val['accuracy'] <= 1, 'Invalid accuracy')
        require(group_edges > 0 if arm == 'digit_full' else group_edges == 0, 'Wrong grouped sampling path')
    left, right = reports['gids'], reports['digit_full']
    require(left['initial_parameters_sha256'] == right['initial_parameters_sha256'], 'Unpaired model initialization')
    require(left['initial_dgl_rng'] == right['initial_dgl_rng'], 'Unpaired initial sampler RNG')
    for a,b in zip(left['epochs'], right['epochs']):
        require(a['root_sha256'] == b['root_sha256'], 'Unpaired root order')
        require(a['validation']['bindings'] == b['validation']['bindings'], 'Different evaluation inputs/features/labels')
    return True


def evidence_directory(data):
    return Path(data)/'native_validation'/sha(ROOT/'ae/release_manifest.json')


def check_native(data):
    directory = evidence_directory(data)
    receipt = directory/'ready.json'
    if not receipt.exists():
        return dict(passed=False, missing=['Paired GIDS/DiGiT native smoke has not passed for this release.'])
    try:
        value = read(receipt)
        require(value['schema'] == 'digit-pa-native-ready-v1' and value['passed'] is True, 'Wrong native receipt')
        binding = current_binding(Path(data))
        require(value['binding'] == binding, 'Native receipt source/data binding changed')
        require(value['returncodes'] == {'gids':0, 'digit_full':0}, 'Native worker did not exit normally')
        reports = {}
        for arm in ('gids', 'digit_full'):
            path = directory/(arm+'.json')
            require(sha(path) == value['reports_sha256'][arm], 'Native report changed: '+arm)
            reports[arm] = read(path)
        validate_reports(reports, binding)
        return dict(passed=True, receipt_sha256=sha(receipt), binding=binding,
                    scope='short native integration only; no full-epoch or convergence claim')
    except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
        return dict(passed=False, missing=['Native receipt rejected: '+str(exc)])
