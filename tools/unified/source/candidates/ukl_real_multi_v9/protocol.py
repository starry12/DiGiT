"""Frozen eight-array loading plans, bounded budgets, and sequential evidence gates."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ukl_real_multi_20261002_v9'
GIB, MIB, PAGE = 1024 ** 3, 1024 ** 2, 4096
STAGE, EXTENT = '64', 64 * GIB
SOURCE = Path('/mnt/n0/digit/ukl_sparse_overlay_v3/full_csc_v1/indices.i32')
NAMES = frozenset(('indptr', 'indices', 'group_ids', 'bases', 'group_ptr', 'covered', 'primary', 'members'))
RESERVE, HEADROOM, CACHE_LIMIT, READ_RATE = 64 * GIB, 8 * GIB, 128 * MIB, 128 * MIB
LOCK = ROOT / 'results/ukl_runtime_prepare_20261001_v6/smoke.lock'
PYTHON = '/home/embed/miniconda3/envs/gids/bin/python'
BASELINE = ROOT / 'results/ukl_real_ladder_20261002_v8/tier_8gib_run_20261002_171516_1469310'
BASELINE_MANIFEST = '39874795e1cc5f83d54c5d7c2883958b788ce991556aeca759b8c0577a4f448f'
RELEASE_FLAGS = ('passed', 'direct_io', 'normal_unregister', 'anonymous_memory_released',
                 'vma_absent_after_release', 'limits_verified_before_cuda')
MULTI_FLAGS = RELEASE_FLAGS + ('source_revalidated_before_cuda', 'source_revalidated_after_release')


def _require(condition, message):
    if not condition:
        raise RuntimeError('Protocol validation failed: ' + message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _duration(value, minimum):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _stage(stage):
    if type(stage) is not str or stage not in ('64', 'full'):
        raise ValueError("Only stages '64' and 'full' are permitted")
    return stage


def identity(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _read_evidence(path):
    _require(not path.is_symlink(), 'symlink evidence: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    _require(len(raw) <= 1024 * 1024, 'oversized evidence: ' + str(path))
    result = json.loads(raw)
    _require(isinstance(result, dict), 'nonobject evidence: ' + str(path))
    return result, hashlib.sha256(raw).hexdigest()


def _source_plan():
    document, _ = _read_evidence(OUT / 'source_plan.json')
    try:
        sources, stages = document['sources'], document['stages']
        _require(isinstance(sources, dict) and set(sources) == NAMES, 'exactly eight named sources required')
        _require(isinstance(stages, dict) and set(stages) == {'64', 'full'}, 'stage set mismatch')
        devices, paths = set(), set()
        for name, source in sources.items():
            ids, path = source['identity'], source['path']
            _require(isinstance(path, str) and Path(path).is_absolute() and '..' not in Path(path).parts,
                     'invalid source path: ' + name)
            _require(isinstance(ids, list) and len(ids) == 5 and all(_integer(v) for v in ids)
                     and _integer(source['bytes'], 1) and ids[2] == source['bytes']
                     and _hash(source['whole_sha256']), 'invalid source identity or digest: ' + name)
            devices.add(ids[0])
            paths.add(path)
        _require(len(devices) == 1 and len(paths) == 8 and sources['indices']['path'] == str(SOURCE),
                 'sources must be distinct files on the bound source device')
        for stage, entry in stages.items():
            specs = entry['specs']
            _require(isinstance(specs, dict) and set(specs) == NAMES, 'stage array set mismatch')
            allocated = logical = 0
            for name, spec in specs.items():
                source = sources[name]
                offset, length = spec['offset'], spec['length']
                _require(spec['path'] == source['path'] and spec['identity'] == source['identity']
                         and _integer(offset) and offset % PAGE == 0 and _integer(length, 1)
                         and offset + length <= source['bytes'], 'range or identity mismatch: ' + name)
                whole = offset == 0 and length == source['bytes']
                if stage == '64':
                    _require(offset + length == source['bytes'], '64 stage range must reach EOF: ' + name)
                    _require((name == 'indices' and not whole) or (name != 'indices' and whole),
                             '64 stage must use seven whole files and an indices suffix')
                if whole:
                    _require(spec.get('sha256') == source['whole_sha256'], 'whole-file digest missing: ' + name)
                else:
                    _require('sha256' not in spec, 'whole-file digest attached to a partial range: ' + name)
                _require(stage != 'full' or whole, 'full stage must include every whole file')
                logical += length
                allocated += (length + PAGE - 1) // PAGE * PAGE
            _require(_integer(entry['arena_bytes'], 1) and entry['arena_bytes'] == allocated
                     and _integer(entry['logical_bytes'], 1) and entry['logical_bytes'] == logical,
                     'stage byte totals differ from the eight ranges')
            _require(stage != '64' or allocated == 64 * GIB, '64 stage must allocate exactly 64 GiB')
            _require(allocated <= 256 * GIB, 'stage exceeds the 256 GiB arena ceiling')
    except (KeyError, TypeError, AttributeError, ValueError) as failure:
        raise RuntimeError('Malformed frozen source plan') from failure
    return document


def plan(stage=None):
    return copy.deepcopy(_source_plan()['stages'][_stage(STAGE if stage is None else stage)])


def configure_stage(stage):
    global STAGE, EXTENT
    selected = _stage(stage)
    extent = plan(selected)['arena_bytes']
    STAGE, EXTENT = selected, extent


def binding():
    document = _source_plan()
    for name, source in document['sources'].items():
        path = Path(source['path'])
        current = path.stat()
        _require(not path.is_symlink() and stat.S_ISREG(current.st_mode)
                 and identity(current) == source['identity'], 'source identity changed: ' + name)
    return copy.deepcopy(document['stages'][_stage(STAGE)]['specs'])


def source_device():
    specs = binding()
    number = specs['indices']['identity'][0]
    device = '%d:%d' % (os.major(number), os.minor(number))
    path = (Path('/dev/block') / device).resolve(strict=True)
    _require(path.is_block_device(), 'source device is not a block device')
    return str(path), device


def budget(stage=None):
    selected = _stage(STAGE if stage is None else stage)
    extent = plan(selected)['arena_bytes']
    maximum = extent + HEADROOM
    return dict(arena_bytes=extent, memory_max=maximum, memory_high=maximum - 512 * MIB,
                memlock=extent + 128 * MIB, host_min=maximum + RESERVE + CACHE_LIMIT,
                own_file_cache_limit=CACHE_LIMIT, read_rate=READ_RATE,
                runtime_seconds=5400 if selected == 'full' else 1800,
                post_seconds=180, quiet_seconds=30, max_quiet_wait_seconds=300,
                full_graph_enabled=False, full_graph_load=selected == 'full')


def full_graph_plan():
    return plan('full')


def verify_manifest():
    manifest = OUT / 'manifest.json'
    values = json.loads(manifest.read_text())
    for name, expected in values.items():
        _require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                 'source/evidence changed: ' + name)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _qualification_valid(result, manifest, boot):
    return (isinstance(result, dict) and result.get('passed') is True and bool(boot)
            and result.get('scope') == 'monitor_only_no_graph_no_cuda'
            and result.get('manifest_sha256') == manifest and result.get('boot_id') == boot
            and _duration(result.get('duration_seconds'), 180)
            and _integer(result.get('samples'), 120)
            and result.get('cgroup_io_stat_observed') is True
            and result.get('gpu_called') is False and result.get('graph_loaded') is False)


def require_monitor_qualification(manifest_sha256):
    result, _ = _read_evidence(OUT / 'monitor_check.json')
    _require(_qualification_valid(result, manifest_sha256, _boot_id()),
             'successful 180-second current-boot monitor qualification required')
    return result


def _state_valid(state, expected):
    return (isinstance(state, dict) and state.get('LoadState') == 'loaded'
            and state.get('MainPID') == '0' and state.get('Result') == 'success'
            and state.get('ExecMainStatus') == '0'
            and (state.get('ActiveState') == 'inactive'
                 or (state.get('ActiveState') == 'active' and state.get('SubState') == 'exited'))
            and state.get('MemoryMax') == str(expected['memory_max'])
            and state.get('MemorySwapMax') == '0'
            and state.get('LimitMEMLOCK') == str(expected['memlock']))


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
            and limits['io_max'][device].get('rbps') == str(READ_RATE))


def validate_worker_report(report, stage=None):
    """Validate receipt data against the frozen plan, without graph or device I/O."""
    selected = _stage(STAGE if stage is None else stage)
    try:
        entry, expected = plan(selected), budget(selected)
        specs = entry['specs']
        _require(isinstance(report, dict) and report.get('stage') == selected
                 and all(report.get(key) is True for key in MULTI_FLAGS)
                 and report.get('full_graph_enabled') is False
                 and report.get('full_graph_load') is (selected == 'full')
                 and report.get('graph_sampling_enabled') is False
                 and report.get('model_enabled') is False
                 and report.get('raw_ssd_access') is False, 'worker scope or release flags invalid')
        for key, value in (('arena_bytes', entry['arena_bytes']), ('logical_bytes', entry['logical_bytes']),
                           ('loaded_bytes', entry['arena_bytes']), ('loaded_payload_bytes', entry['logical_bytes']),
                           ('gpu_readback_bytes', entry['logical_bytes']),
                           ('gpu_readback_allocated_bytes', entry['arena_bytes']),
                           ('padding_bytes', entry['arena_bytes'] - entry['logical_bytes'])):
            _require(_integer(report.get(key)) and report[key] == value, 'worker byte total mismatch: ' + key)
        arrays = report['arrays']
        _require(isinstance(arrays, dict) and set(arrays) == NAMES, 'worker array set mismatch')
        arena_offset = 0
        for name, spec in specs.items():
            row = arrays[name]
            allocated = (spec['length'] + PAGE - 1) // PAGE * PAGE
            _require(isinstance(row, dict) and row.get('path') == spec['path']
                     and row.get('source_identity') == spec['identity']
                     and _integer(row.get('offset')) and row['offset'] == spec['offset']
                     and _integer(row.get('logical_length'), 1) and row['logical_length'] == spec['length']
                     and _integer(row.get('allocated_bytes'), 1)
                     and row['allocated_bytes'] == allocated
                     and _integer(row.get('arena_offset')) and row['arena_offset'] == arena_offset
                     and _integer(row.get('padding_bytes')) and row['padding_bytes'] == allocated - spec['length'],
                     'worker array range mismatch: ' + name)
            arena_offset += allocated
            _require(_hash(row.get('cpu_sha256')) and row['cpu_sha256'] == row.get('gpu_sha256')
                     and row.get('cpu_padding_zero') is True and row.get('gpu_padding_zero') is True,
                     'worker array hash or padding mismatch: ' + name)
            wanted = spec.get('sha256')
            _require(row.get('expected_sha256') == wanted
                     and (row.get('expected_sha256_matched') is True and row['cpu_sha256'] == wanted
                          if wanted is not None else row.get('expected_sha256_matched') is None),
                     'whole-file integrity mismatch: ' + name)
        number = specs['indices']['identity'][0]
        device = '%d:%d' % (os.major(number), os.minor(number))
        _require(_limits_valid(report.get('effective_limits'), expected, device), 'worker effective limits invalid')
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return True


def worker_passed(state, report, stage=None):
    selected = _stage(STAGE if stage is None else stage)
    try:
        return _state_valid(state, budget(selected)) and validate_worker_report(report, selected)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _baseline_manifest_digest():
    from candidates.ukl_real_ladder_v8 import protocol as previous
    return previous.verify_manifest()


def _baseline_budget():
    from candidates.ukl_real_ladder_v8 import protocol as previous
    return previous.budget(8)


def _newest_64_attempt():
    attempts = []
    for path in OUT.glob('stage_64_run_*'):
        match = re.fullmatch(r'stage_64_run_([0-9]{8})_([0-9]{6})_([0-9]+)', path.name)
        _require(match is not None and path.is_dir() and not path.is_symlink(),
                 'unexpected predecessor attempt: ' + str(path))
        attempts.append(((match[1], match[2], int(match[3])), path))
    _require(bool(attempts), 'a completed 64 GiB stage is required before full')
    return max(attempts)[1]


def _acceptance_valid(accepted, worker, before, manifest, boot):
    _require(accepted.get('passed') is True and accepted.get('worker_launched') is True
             and accepted.get('error') is None and accepted.get('worker') == worker
             and accepted.get('manifest_sha256') == manifest and before.get('manifest_sha256') == manifest,
             'predecessor failed, incomplete, or inconsistent')
    _require(_qualification_valid(before.get('monitor_qualification'), manifest, boot),
             'predecessor monitor qualification or boot differs')
    _require(accepted.get('kernel_monitor_ok') is True and accepted.get('post_guard_passed') is True
             and _duration(accepted.get('post_seconds'), 180)
             and accepted.get('io_preflight_passed') is True
             and _integer(accepted.get('runtime_valid_samples'), 1)
             and accepted.get('full_graph_enabled') is False and accepted.get('raw_ssd_access') is False,
             'predecessor kernel, post-exit, or runtime evidence incomplete')
    probes = accepted.get('post_probes')
    _require(isinstance(probes, list) and len(probes) == 2
             and all(isinstance(row, dict) and row.get('passed') is True
                     and _integer(row.get('exit_code')) and row['exit_code'] == 0 for row in probes)
             and {str(Path(row['path']).parent) for row in probes} == {str(ROOT / 'results'), '/mnt/n0'},
             'predecessor filesystem probes invalid')
    _require(isinstance(accepted.get('unit'), str) and bool(accepted['unit'])
             and worker['effective_limits']['cgroup'] == '/system.slice/' + accepted['unit'],
             'predecessor worker cgroup differs')


def require_predecessor():
    """Admit 64 from the fixed accepted 8 GiB run; full from newest accepted 64."""
    selected = _stage(STAGE)
    manifest = verify_manifest()
    binding()  # Revalidate every current file even for the single-file bootstrap.
    sources = _source_plan()['sources']
    baseline = selected == '64'
    folder = BASELINE if baseline else _newest_64_attempt()
    expected_manifest = BASELINE_MANIFEST if baseline else manifest
    if baseline:
        _require(_baseline_manifest_digest() == BASELINE_MANIFEST, 'frozen v8 manifest changed')
    try:
        documents, hashes = {}, {}
        for name in ('acceptance.json', 'worker.json', 'before.json'):
            documents[name], hashes[name] = _read_evidence(folder / name)
        accepted, worker, before = (documents[name] for name in ('acceptance.json', 'worker.json', 'before.json'))
        boot = _boot_id()
        _acceptance_valid(accepted, worker, before, expected_manifest, boot)
        if baseline:
            expected = _baseline_budget()
            source = sources['indices']
            number = source['identity'][0]
            device = '%d:%d' % (os.major(number), os.minor(number))
            _require(all(type(document.get('tier_gib')) is int and document['tier_gib'] == 8
                         for document in (accepted, worker, before)), 'bootstrap tier differs from 8 GiB')
            _require(_state_valid(accepted['worker_state'], expected)
                     and all(worker.get(key) is True for key in RELEASE_FLAGS)
                     and all(_integer(worker.get(key)) and worker[key] == 8 * GIB for key in
                             ('extent_bytes', 'loaded_bytes', 'gpu_readback_bytes'))
                     and _hash(worker.get('source_sha256')) and worker['source_sha256'] == worker.get('gpu_sha256')
                     and worker.get('source_identity') == source['identity']
                     and worker.get('full_graph_enabled') is False
                     and _limits_valid(worker.get('effective_limits'), expected, device),
                     'bootstrap lifecycle, hash, source identity, or limits invalid')
            old = before['source']
            _require(old.get('path') == source['path'] and old.get('identity') == source['identity']
                     and type(old.get('offset')) is int and old['offset'] == 0
                     and type(old.get('length')) is int and old['length'] == 8 * GIB,
                     'bootstrap source range differs')
            prior = before.get('predecessor')
            _require(isinstance(prior, dict) and prior.get('passed') is True
                     and prior == accepted.get('predecessor') and prior.get('target_tier_gib') == 8
                     and prior.get('predecessor_tier_gib') == 4, 'bootstrap predecessor chain differs')
        else:
            expected = budget('64')
            _require(accepted.get('stage') == '64' and before.get('stage') == '64'
                     and accepted.get('full_graph_load') is False
                     and before.get('source') == plan('64')['specs']
                     and worker_passed(accepted['worker_state'], worker, '64'),
                     '64 GiB predecessor scope, sources, or worker receipt invalid')
            prior = before.get('predecessor')
            _require(isinstance(prior, dict) and prior.get('passed') is True
                     and prior == accepted.get('predecessor') and prior.get('target_stage') == '64'
                     and prior.get('predecessor_stage') == '8gib'
                     and prior.get('output') == str(BASELINE)
                     and prior.get('manifest_sha256') == BASELINE_MANIFEST,
                     '64 GiB saved predecessor differs from the fixed bootstrap')
        _require(isinstance(before.get('budget'), dict)
                 and all(before['budget'].get(key) == value for key, value in expected.items()),
                 'predecessor saved budget differs')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as failure:
        raise RuntimeError('Predecessor evidence missing or malformed: ' + str(folder)) from failure
    return dict(passed=True, target_stage=selected, predecessor_stage='8gib' if baseline else '64',
                output=str(folder), manifest_sha256=expected_manifest, boot_id=boot,
                source_identities={name: value['identity'] for name, value in sources.items()},
                evidence_sha256=hashes)
