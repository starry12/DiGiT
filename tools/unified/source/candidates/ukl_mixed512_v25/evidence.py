"""Require both accepted full-graph smoke exits, not computation checkpoints."""
import hashlib
import json
from pathlib import Path


def accepted_smoke(root, binding_path, storage):
    binding = json.loads(binding_path.read_text())
    if set(binding['receipts']) != {'pair', 'gids', 'digit'}:
        raise RuntimeError('Both full-graph smoke arms required')
    records = {}
    for key, entry in binding['receipts'].items():
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Invalid smoke receipt path')
        p = root / rel
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Smoke receipt changed: '+key)
        records[key] = json.loads(p.read_text())
    pair = records['pair']
    if not (pair['passed'] is True and pair['batches_per_arm'] == 4
            and pair['performance_run'] is False and pair['arms'] == ['gids', 'digit']):
        raise RuntimeError('Smoke pair not accepted')
    for arm in ('gids', 'digit'):
        a = records[arm]; w = a['worker']; state = a['worker_state']
        checks = [a['passed'] is True, a['stage'] == arm,
                  a['pair_id'] == pair['pair_id'], a['selected_gpu'] == pair['selected_gpu'],
                  a['manifest_sha256'] == binding['original_manifest_sha256'],
                  a['kernel_monitor_ok'] is True, a['post_guard_passed'] is True,
                  a['gpu_released'] is True, a['full_graph_load'] is True,
                  state['MainPID'] == '0', state['Result'] == 'success',
                  state['ExecMainStatus'] == '0', w['passed'] is True,
                  w['updates'] == 4, w['mode'] == 'smoke', w['finite'] is True,
                  w['external_cache_released'] is True, w['graph_and_aux_released'] is True,
                  w['storage_acceptance_sha256'] == storage['acceptance_sha256'],
                  a['predecessor']['source_binding_sha256'] == storage['source_binding_sha256'],
                  w['ownership']['released'] is True,
                  all(x['passed'] is True for x in w['feature_value_checks']),
                  all(x['reconciled'] is True for x in w['counters'].values()),
                  len(a['post_probes']) == 2 and all(x['passed'] is True for x in a['post_probes'])]
        if not all(checks):
            raise RuntimeError('Full-graph smoke arm not accepted: '+arm)
    return dict(pair_id=pair['pair_id'], receipts=binding['receipts'])
