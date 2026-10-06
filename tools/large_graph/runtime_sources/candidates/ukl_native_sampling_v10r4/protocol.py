"""Frozen UKL sources, v9 admission, and bounded native sampler acceptance.

All functions in this module read metadata and JSON receipts only. Graph payloads,
CUDA calls, and service management belong to the admitted worker/controller.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from candidates.ukl_real_multi_v9 import protocol as V9
from candidates.ukl_native_sampling_v10 import protocol as V10

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ukl_native_sampling_20261002_v10/repair4'
GIB, MIB, PAGE = V9.GIB, V9.MIB, V9.PAGE
STAGE, EXTENT = 'sampling', 224647307264
LOCK, PYTHON = V9.LOCK, V9.PYTHON
NAMES = V9.NAMES
RESERVE, HEADROOM, CACHE_LIMIT, READ_RATE = 64 * GIB, 12 * GIB, 128 * MIB, GIB
APPLICATION_READ_RATE = 512 * MIB
FROZEN_V10_MANIFEST = '4661076401ccc5a4917c40ca0a8fa47d65c14273d5d889f2f8eb803f772f5c07'
BASELINE_MANIFEST = '5e59d01cc84010cd34cd95f96bfad3a1ca2f8b73b11f775e7521ebc31648fc5b'
BASELINE = V9.OUT / 'stage_full_run_20261002_180048_1494808'
RELEASE_FLAGS, MULTI_FLAGS = V9.RELEASE_FLAGS, V9.MULTI_FLAGS
ARMS = ('gids', 'digit')
FANOUTS, BATCHES, ROOTS = (10, 5, 5), 4, 1024
METADATA = dict(nodes=787801471, edges=48071979976, groups=373670588,
                units=373670588, storage_rows=945361765, origin=0, unit_origin=0)
_require, _integer, _hash = V9._require, V9._integer, V9._hash
_duration, _state_valid = V9._duration, V9._state_valid
_qualification_valid = V9._qualification_valid
_read_evidence, _boot_id, identity = V9._read_evidence, V9._boot_id, V9.identity


def _limits_valid(limits, expected, device):
    if not isinstance(limits, dict):
        return False
    for key, value in (('memory.max', expected['memory_max']), ('memory.high', expected['memory_high']),
                       ('memory.swap.max', 0), ('pids.max', 64),
                       ('memlock_soft', expected['memlock']), ('memlock_hard', expected['memlock'])):
        if not _integer(limits.get(key)) or limits[key] != value:
            return False
    return (_integer(limits.get('cpu_quota'), 1) and _integer(limits.get('cpu_period'), 1)
            and limits['cpu_quota'] <= limits['cpu_period']
            and isinstance(limits.get('cgroup'), str) and limits['cgroup'].startswith('/system.slice/')
            and isinstance(limits.get('io_max'), dict)
            and isinstance(limits['io_max'].get(device), dict)
            and limits['io_max'][device].get('rbps') == str(expected['read_rate']))


def _stage(stage):
    if type(stage) is not str or stage != 'sampling':
        raise ValueError("Only stage 'sampling' is permitted")
    return stage


def _int_sequence(value, expected):
    return isinstance(value, list) and value == list(expected) and all(type(v) is int for v in value)


def plan(stage=None):
    _stage(STAGE if stage is None else stage)
    return V9.plan('full')


def metadata():
    """Check every frozen logical array shape without opening graph payloads."""
    specs = plan()['specs']
    m = METADATA
    shapes = dict(indptr=8 * (m['nodes'] + 1), indices=4 * m['edges'],
                  group_ids=4 * m['units'], bases=8 * m['groups'],
                  group_ptr=8 * (m['nodes'] + 1), covered=16 * m['units'],
                  primary=8 * m['nodes'], members=8 * m['groups'])
    _require(set(specs) == NAMES, 'metadata requires exactly eight arrays')
    for name, length in shapes.items():
        spec = specs[name]
        _require(type(spec.get('length')) is int and spec['length'] == length
                 and type(spec.get('offset')) is int and spec['offset'] == 0,
                 'frozen logical shape differs: ' + name)
    return dict(m)


def configure_stage(stage):
    global STAGE, EXTENT
    selected = _stage(stage)
    extent = plan(selected)['arena_bytes']
    metadata()
    STAGE, EXTENT = selected, extent


def binding():
    specs = plan()['specs']
    metadata()
    for name, spec in specs.items():
        path = Path(spec['path'])
        current = path.stat()
        _require(not path.is_symlink() and stat.S_ISREG(current.st_mode)
                 and identity(current) == spec['identity'], 'source identity changed: ' + name)
    return copy.deepcopy(specs)


def source_device():
    number = binding()['indices']['identity'][0]
    device = '%d:%d' % (os.major(number), os.minor(number))
    path = (Path('/dev/block') / device).resolve(strict=True)
    _require(path.is_block_device(), 'source device is not a block device')
    return str(path), device


def budget(stage=None):
    extent = plan(_stage(STAGE if stage is None else stage))['arena_bytes']
    maximum = extent + HEADROOM
    return dict(arena_bytes=extent, memory_max=maximum, memory_high=maximum - 512 * MIB,
                memlock=extent + 128 * MIB, host_min=maximum + RESERVE + CACHE_LIMIT,
                own_file_cache_limit=CACHE_LIMIT, read_rate=READ_RATE, application_read_rate=APPLICATION_READ_RATE, runtime_seconds=5400,
                post_seconds=180, quiet_seconds=30, max_quiet_wait_seconds=300,
                full_graph_enabled=True, full_graph_load=True, graph_sampling_enabled=True,
                model_enabled=False, features_enabled=False, raw_ssd_access=False)


def verify_manifest():
    _require(V9.verify_manifest() == BASELINE_MANIFEST, 'frozen v9 manifest changed')
    _require(V10.verify_manifest() == FROZEN_V10_MANIFEST, 'frozen v10 manifest changed')
    manifest = OUT / 'manifest.json'
    values, digest = _read_evidence(manifest)
    _require(bool(values), 'empty v10 repair manifest')
    for name, expected in values.items():
        path = Path(name)
        _require(not path.is_absolute() and '..' not in path.parts and _hash(expected),
                 'invalid manifest entry: ' + name)
        path = ROOT / path
        _require(not path.is_symlink() and path.is_file()
                 and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                 'source/evidence changed: ' + name)
    return digest


def require_monitor_qualification(manifest_sha256):
    result, _ = _read_evidence(OUT / 'monitor_check.json')
    _require(_qualification_valid(result, manifest_sha256, _boot_id()),
             'successful 180-second current-boot monitor qualification required')
    return result


def _newest_full_attempt():
    attempts = []
    for path in V9.OUT.glob('stage_full_run_*'):
        match = re.fullmatch(r'stage_full_run_([0-9]{8})_([0-9]{6})_([0-9]+)', path.name)
        _require(match is not None and path.is_dir() and not path.is_symlink(),
                 'unexpected v9 full attempt: ' + str(path))
        attempts.append(((match[1], match[2], int(match[3])), path))
    _require(bool(attempts), 'accepted v9 full attempt is required')
    return max(attempts)[1]


def _v9_predecessor():
    """Recheck the existing 64 GiB gate and restore v9 module configuration."""
    stage, extent = V9.STAGE, V9.EXTENT
    try:
        V9.configure_stage('full')
        return V9.require_predecessor()
    finally:
        V9.STAGE, V9.EXTENT = stage, extent


def require_predecessor():
    """Require this accepted full attempt plus its unchanged current v9 gate."""
    _stage(STAGE)
    verify_manifest()
    specs = binding()
    _require(V9.verify_manifest() == BASELINE_MANIFEST, 'frozen v9 manifest changed')
    folder = _newest_full_attempt()
    _require(folder == BASELINE, 'latest v9 full attempt differs from the fixed accepted run')
    try:
        documents, hashes = {}, {}
        for name in ('acceptance.json', 'worker.json', 'before.json'):
            documents[name], hashes[name] = _read_evidence(folder / name)
        accepted, worker, before = (documents[name] for name in ('acceptance.json', 'worker.json', 'before.json'))
        boot = _boot_id()
        V9._acceptance_valid(accepted, worker, before, BASELINE_MANIFEST, boot)
        _require(accepted.get('stage') == before.get('stage') == 'full'
                 and accepted.get('full_graph_load') is True
                 and before.get('source') == specs
                 and V9.worker_passed(accepted['worker_state'], worker, 'full'),
                 'v9 full scope, source, worker, or release receipt invalid')
        _require(before.get('budget') == V9.budget('full'), 'v9 saved budget differs')
        prior = _v9_predecessor()
        _require(prior == before.get('predecessor') == accepted.get('predecessor'),
                 'v9 saved 64 GiB predecessor gate changed')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as failure:
        raise RuntimeError('Predecessor evidence missing or malformed: ' + str(folder)) from failure
    return dict(passed=True, target_stage='sampling', predecessor_stage='full',
                output=str(folder), manifest_sha256=BASELINE_MANIFEST, boot_id=boot,
                source_identities={name: row['identity'] for name, row in specs.items()},
                evidence_sha256=hashes, predecessor=prior)


def _validate_raw_matrix(receipts, roots):
    """Require every arm/fanout/seed case once, with bounded matching outputs."""
    expected = {(arm, fanout, seed) for arm in ARMS
                for fanout in (1, 5, 10, 32) for seed in (0, 23)}
    _require(isinstance(receipts, list) and len(receipts) == len(expected),
             'sixteen detailed raw comparisons required')
    observed = set()
    for row in receipts:
        _require(isinstance(row, dict) and isinstance(row.get('arm'), str)
                 and type(row.get('fanout')) is int and type(row.get('seed')) is int,
                 'raw comparison case identity invalid')
        case = (row['arm'], row['fanout'], row['seed'])
        _require(case in expected and case not in observed,
                 'raw comparison case missing, duplicated, or outside the matrix')
        observed.add(case)
        _require(_integer(row.get('edges')) and row['edges'] <= roots * row['fanout']
                 and _hash(row.get('raw_sha256'))
                 and row.get('cpu_raw_sha256') == row['raw_sha256'],
                 'raw comparison count or CPU/GPU digest invalid')
    _require(observed == expected, 'raw comparison coverage incomplete')


def _validate_sampling(sampling):
    _require(isinstance(sampling, dict) and sampling.get('passed') is True,
             'sampling evidence missing or failed')
    synthetic = sampling['synthetic']
    _require(isinstance(synthetic, dict)
             and all(synthetic.get(k) is True for k in
                     ('passed', 'high_eid_exercised', 'digit_packed', 'normal_release'))
             and synthetic.get('baseline_packed') is False
             and type(synthetic.get('cases')) is int and synthetic['cases'] == 2
             and type(synthetic.get('draws')) is int and synthetic['draws'] == 32,
             'synthetic sampler matrix incomplete')
    fixtures = synthetic['receipts']
    _require(isinstance(fixtures, list) and len(fixtures) == 2,
             'two detailed synthetic fixture receipts required')
    for high_origin, fixture in zip((False, True), fixtures):
        _require(isinstance(fixture, dict) and fixture.get('high_origin') is high_origin
                 and fixture.get('vma_absent') is True
                 and _integer(fixture.get('arena_bytes'), 8 * PAGE)
                 and fixture['arena_bytes'] <= MIB and fixture['arena_bytes'] % PAGE == 0,
                 'synthetic fixture origin, release, or bounded arena invalid')
        _validate_raw_matrix(fixture.get('draws'), 6)
    _require(type(synthetic.get('peak_arena_bytes')) is int
             and synthetic['peak_arena_bytes'] == max(row['arena_bytes'] for row in fixtures),
             'synthetic peak allocation differs from detailed receipts')
    single = sampling['single_layer']
    _require(isinstance(single, dict)
             and all(single.get(k) is True for k in
                     ('passed', 'high_eid_exercised', 'raw_parity', 'address_checks'))
             and type(single.get('cases')) is int and single['cases'] == 16
             and _int_sequence(single.get('fanouts'), (1, 5, 10, 32))
             and _int_sequence(single.get('seeds'), (0, 23)),
             'full graph single-layer matrix incomplete')
    _require(_integer(single.get('root_count'), 1) and single['root_count'] <= 8
             and _hash(single.get('roots_sha256')), 'small full-graph root receipt invalid')
    _validate_raw_matrix(single.get('receipts'), single['root_count'])
    arms = sampling['arms']
    _require(isinstance(arms, dict) and set(arms) == set(ARMS), 'sampling arms differ')
    for arm in ARMS:
        batches = arms[arm]
        _require(isinstance(batches, list) and len(batches) == BATCHES, 'four batches required per arm')
        for batch_index, batch in enumerate(batches):
            _require(isinstance(batch, dict)
                     and type(batch.get('batch')) is int and batch['batch'] == batch_index
                     and type(batch.get('root_count')) is int and batch['root_count'] == ROOTS
                     and type(batch.get('layer_count')) is int and batch['layer_count'] == 3
                     and _int_sequence(batch.get('fanouts'), FANOUTS)
                     and _hash(batch.get('roots_sha256'))
                     and all(batch.get(k) is True for k in ('cpu_gpu_raw_parity', 'frontier_parity',
                         'eid_parity', 'block_parity', 'address_checks', 'storage_mapping')),
                     'batch identity, dimensions, or parity incomplete')
            layers = batch['layers']
            _require(isinstance(layers, list) and len(layers) == 3, 'three layer receipts required')
            destinations = ROOTS
            # DGL block order is opposite to the order in which roots expand.
            for index in (2, 1, 0):
                row = layers[index]
                fanout = FANOUTS[index]
                _require(isinstance(row, dict) and type(row.get('layer')) is int and row['layer'] == index
                         and type(row.get('fanout')) is int and row['fanout'] == fanout
                         and type(row.get('root_count')) is int and row['root_count'] == destinations
                         and _integer(row.get('edge_count')) and row['edge_count'] <= destinations * fanout
                         and _integer(row.get('frontier_count'), destinations)
                         and row['frontier_count'] <= destinations + row['edge_count'],
                         'layer counts or frontier chain invalid')
                for name in ('raw', 'frontier', 'eid', 'row', 'block', 'storage'):
                    key = name + '_sha256'
                    _require(_hash(row.get(key)) and row.get('cpu_' + key) == row[key],
                             'CPU/GPU layer hash differs: ' + name)
                destinations = row['frontier_count']
            _require(type(batch.get('input_nodes_count')) is int
                     and batch['input_nodes_count'] == layers[0]['frontier_count']
                     and batch.get('input_nodes_sha256') == layers[0]['frontier_sha256'],
                     'input node receipt differs from outermost frontier')
        _require(len({b['roots_sha256'] for b in batches}) == BATCHES,
                 'the four batches must have distinct roots')
    _require([b['roots_sha256'] for b in arms['gids']] == [b['roots_sha256'] for b in arms['digit']],
             'GIDS and DiGiT batches must use the same roots')


REPAIR_FLAGS = ('cpu_framework_prewarmed_before_any_registration',
                'nofork_guard_before_any_arena', 'nofork_guard_held_through_release',
                'nofork_guard_released_after_all_arenas', 'all_arenas_dontfork_verified')


def _dontfork_valid(receipt, extent):
    """Require page-aligned anonymous VMA coverage, not just a summary flag."""
    _require(isinstance(receipt, dict)
             and all(receipt.get(k) is True for k in
                     ('madv_dontfork', 'vmflags_dc', 'anonymous_private')),
             'anonymous DONTFORK attestation missing')
    address = receipt.get('address')
    _require(_integer(address, PAGE) and address % PAGE == 0
             and type(receipt.get('bytes')) is int and receipt['bytes'] == extent
             and type(receipt.get('verified_bytes')) is int and receipt['verified_bytes'] == extent,
             'DONTFORK range differs from owned arena')
    vmas = receipt.get('vmas')
    _require(isinstance(vmas, list) and 1 <= len(vmas) <= 4096,
             'bounded DONTFORK VMA evidence required')
    cursor = address
    for row in vmas:
        _require(isinstance(row, dict) and _integer(row.get('start'))
                 and _integer(row.get('end'), PAGE)
                 and row['start'] % PAGE == row['end'] % PAGE == 0
                 and row['end'] > row['start'], 'invalid DONTFORK VMA bounds')
        flags = row.get('vmflags')
        _require(isinstance(flags, list) and 'dc' in flags
                 and all(isinstance(flag, str) and 1 <= len(flag) <= 8 for flag in flags)
                 and 'sh' not in flags, 'VMA lacks private DONTFORK flags')
        start, end = max(address, row['start']), min(address + extent, row['end'])
        _require(start == cursor and end > start, 'DONTFORK VMA gap or overlap')
        cursor = end
    _require(cursor == address + extent, 'DONTFORK VMA coverage incomplete')
    return address


def _validate_repair(report, extent):
    _require(all(report.get(key) is True for key in REPAIR_FLAGS),
             'prewarm, no-fork, or anonymous ownership repair not attested')
    prewarm = report.get('cpu_framework_prewarm')
    _require(isinstance(prewarm, dict)
             and all(prewarm.get(key) is True for key in
                     ('passed', 'cpu_only', 'before_ownership'))
             and prewarm.get('cuda_initialized_before') is False
             and prewarm.get('cuda_initialized_after') is False
             and type(prewarm.get('cpu_blocks')) is int and prewarm['cpu_blocks'] == 2
             and _integer(prewarm.get('pid'), 1), 'CPU framework prewarm receipt invalid')
    for key in ('torch_version', 'dgl_version'):
        _require(isinstance(prewarm.get(key), str) and 1 <= len(prewarm[key]) <= 128,
                 'prewarmed framework version missing')
    for key, active, released in (('nofork_guard_receipt', True, False),
                                  ('nofork_guard_release_receipt', False, True)):
        guard = report.get(key)
        _require(isinstance(guard, dict) and guard.get('active') is active
                 and guard.get('released') is released
                 and guard.get('audit_hook_verified') is True
                 and type(guard.get('entered_pid')) is int
                 and guard['entered_pid'] == prewarm['pid'],
                 'no-fork guard identity or lifecycle invalid')
        for count, expected in (('blocked_events', 0), ('tracked_arenas', 3), ('released_arenas', 3)):
            _require(type(guard.get(count)) is int and guard[count] == expected,
                     'no-fork guard count differs: ' + count)
    before = _dontfork_valid(report.get('full_arena_dontfork_before_load'), extent)
    after = _dontfork_valid(report.get('full_arena_dontfork_before_registration'), extent)
    registered = _dontfork_valid(report.get('full_arena_dontfork_after_registration'), extent)
    _require(before == after == registered, 'full arena address or DONTFORK changed across registration')
    for fixture in report['sampling']['synthetic']['receipts']:
        before = _dontfork_valid(fixture.get('dontfork_before_load'), fixture['arena_bytes'])
        after = _dontfork_valid(fixture.get('dontfork_before_registration'), fixture['arena_bytes'])
        registered = _dontfork_valid(fixture.get('dontfork_after_registration'), fixture['arena_bytes'])
        _require(before == after == registered, 'synthetic arena address or DONTFORK changed across registration')


def validate_worker_report(report, stage=None):
    """Check bounded JSON sampling evidence and known logical SHA256 values."""
    selected = _stage(STAGE if stage is None else stage)
    try:
        entry, expected = plan(selected), budget(selected)
        _require(report.get('application_read_rate') == APPLICATION_READ_RATE
                 and report.get('hard_read_rate') == READ_RATE, 'read pacing differs')
        specs = entry['specs']
        _require(isinstance(report, dict) and len(json.dumps(report, allow_nan=False)) <= MIB,
                 'worker report must be bounded JSON')
        _require(report.get('stage') == selected and all(report.get(k) is True for k in MULTI_FLAGS),
                 'worker stage or lifecycle invalid')
        for key, value in (('full_graph_enabled', True), ('full_graph_load', True),
                           ('graph_sampling_enabled', True), ('model_enabled', False),
                           ('features_enabled', False), ('raw_ssd_access', False)):
            _require(report.get(key) is value, 'worker scope invalid: ' + key)
        _require(report.get('metadata') == metadata()
                 and all(type(v) is int for v in report['metadata'].values()),
                 'worker logical metadata differs')
        for key, value in (('arena_bytes', entry['arena_bytes']), ('logical_bytes', entry['logical_bytes']),
                           ('loaded_bytes', entry['arena_bytes']), ('loaded_payload_bytes', entry['logical_bytes']),
                           ('padding_bytes', entry['arena_bytes'] - entry['logical_bytes'])):
            _require(_integer(report.get(key)) and report[key] == value, 'worker byte total differs: ' + key)
        arrays = report['arrays']
        _require(isinstance(arrays, dict) and set(arrays) == NAMES, 'worker array set differs')
        arena_offset = 0
        for name, spec in specs.items():
            row = arrays[name]
            allocated = (spec['length'] + PAGE - 1) // PAGE * PAGE
            _require(isinstance(row, dict) and row.get('path') == spec['path']
                     and row.get('source_identity') == spec['identity'], 'worker array identity differs: ' + name)
            for key, value in (('offset', spec['offset']), ('logical_length', spec['length']),
                               ('allocated_bytes', allocated), ('arena_offset', arena_offset),
                               ('padding_bytes', allocated - spec['length'])):
                _require(_integer(row.get(key)) and row[key] == value, 'worker array range differs: ' + name)
            arena_offset += allocated
            _require(_hash(spec.get('sha256')) and row.get('cpu_sha256') == spec['sha256']
                     and row.get('expected_sha256') == spec['sha256']
                     and row.get('expected_sha256_matched') is True
                     and row.get('cpu_padding_zero') is True, 'whole-file integrity differs: ' + name)
        number = specs['indices']['identity'][0]
        device = '%d:%d' % (os.major(number), os.minor(number))
        _require(_limits_valid(report.get('effective_limits'), expected, device), 'worker effective limits invalid')
        _validate_sampling(report.get('sampling'))
        _validate_repair(report, entry['arena_bytes'])
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return False
    return True


def worker_passed(state, report, stage=None):
    selected = _stage(STAGE if stage is None else stage)
    try:
        return _state_valid(state, budget(selected)) and validate_worker_report(report, selected)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
