"""Reuse frozen successful SSD preloads; never relabel them as current runs."""
import hashlib
import json
from pathlib import Path


def accepted_preloads(root, binding_path, storage):
    binding = json.loads(binding_path.read_text())
    if set(binding['receipts']) != {'ssd_gids', 'ssd_digit'}:
        raise RuntimeError('Both previously accepted SSD preloads required')
    result = {}
    for stage, entry in binding['receipts'].items():
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Invalid preload evidence path')
        path = root / rel
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Reused preload evidence changed: '+stage)
        a = json.loads(path.read_text()); w = a['worker']; state = a['worker_state']
        io = w['io']; ownership = w['ownership']
        expected_arenas = 1 if stage == 'ssd_digit' else 0
        checks = (
            a['passed'] is True, a['stage'] == stage,
            a['kernel_monitor_ok'] is True, a['post_guard_passed'] is True,
            a['gpu_released'] is True, a['raw_ssd_writes'] is False,
            a['manifest_sha256'] == binding['original_manifest_sha256'],
            state['MainPID'] == '0', state['Result'] == 'success',
            state['ExecMainStatus'] == '0', state['SubState'] in ('exited', 'dead'),
            w['passed'] is True, w['mode'] == 'ssd_preload', w['stage'] == stage,
            w['arm'] == stage.replace('ssd_', ''), w['rows'] == 65536,
            w['values_checked'] == 65536*128, w['updates'] == 0,
            w['cache_released'] is True, w['graph_loaded'] is False,
            w['effective_limits_verified'] is True, w['raw_ssd_writes'] is False,
            w['manifest_sha256'] == a['manifest_sha256'],
            w['pair_id'] == a['pair_id'], w['selected_gpu'] == a['selected_gpu'],
            w['storage_acceptance_sha256'] == storage['acceptance_sha256'],
            a['predecessor']['source_binding_sha256'] == storage['source_binding_sha256'],
            ownership['released'] is True, ownership['active'] is False,
            ownership['tracked_arenas'] == ownership['released_arenas'] == expected_arenas,
            io['submitted_commands'] > 0,
            io['submitted_commands'] == io['completed_commands'],
            io['outstanding'] == 0, io['completed_bytes'] > 0)
        if not all(checks):
            raise RuntimeError('Reused SSD preload is not accepted: '+stage)
        result[stage] = dict(**entry, reused=True, original_pair_id=a['pair_id'],
                             original_gpu=a['selected_gpu'])
    return result
