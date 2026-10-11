"""Fixed 2/4/8 GiB real-file tiers with evidence-gated sequential admission."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ukl_real_ladder_20261002_v8'
GIB = 1024**3
MIB = 1024**2
TIER = 2
EXTENT = TIER * GIB
SOURCE = Path('/mnt/n0/digit/ukl_sparse_overlay_v3/full_csc_v1/indices.i32')
RESERVE = 64 * GIB
HEADROOM = 3 * GIB
CACHE_LIMIT = 128 * MIB
READ_RATE = 128 * MIB
LOCK = ROOT / 'results/ukl_runtime_prepare_20261001_v6/smoke.lock'
PYTHON = '/home/embed/miniconda3/envs/gids/bin/python'
BASELINE = ROOT / 'results/ukl_guarded_loader_20261002_v7/repair3/run_20261002_163651_1456014'
BASELINE_MANIFEST = 'bb2b10fcc22d1593da2a8dbded11244245cf6274e120b525efb6c16ce3c2d7e2'


def _tier(tier):
    if type(tier) is not int or tier not in (2, 4, 8):
        raise ValueError('Only integer tiers 2, 4, and 8 GiB are permitted')
    return tier


def configure_tier(tier):
    global TIER, EXTENT
    TIER = _tier(tier)
    EXTENT = TIER * GIB


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def identity(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def verify_manifest():
    manifest = OUT / 'manifest.json'
    values = json.loads(manifest.read_text())
    for name, want in values.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != want:
            raise RuntimeError('Source/evidence changed: ' + name)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def require_monitor_qualification(manifest_sha256):
    """Require the repaired monitor to have run on this boot with these sources."""
    path = OUT / 'monitor_check.json'
    result = json.loads(path.read_text())
    boot = _boot_id()
    if (result.get('passed') is not True
            or result.get('scope') != 'monitor_only_no_graph_no_cuda'
            or result.get('manifest_sha256') != manifest_sha256
            or result.get('boot_id') != boot
            or result.get('duration_seconds', 0) < 180
            or result.get('samples', 0) < 120
            or result.get('cgroup_io_stat_observed') is not True
            or result.get('gpu_called') is not False
            or result.get('graph_loaded') is not False):
        raise RuntimeError('A successful 180-second monitor qualification is required on this boot')
    return result


def binding():
    d = json.loads((OUT / 'source_binding.json').read_text())
    _tier(TIER)
    if (d['path'] != str(SOURCE) or type(d['offset']) is not int or d['offset'] != 0
            or type(d['length']) is not int or d['length'] != GIB):
        raise RuntimeError('The original fixed 1 GiB source binding is required')
    source = SOURCE.stat()
    if (SOURCE.is_symlink() or not stat.S_ISREG(source.st_mode)
            or identity(source) != d['identity'] or source.st_size < EXTENT):
        raise RuntimeError('Graph source identity changed')
    return dict(d, length=EXTENT)


def budget(tier=None):
    # Direct I/O avoids charging a second graph-sized page cache to this worker.
    tier = _tier(TIER if tier is None else tier)
    extent = tier * GIB
    maximum = extent + 3 * GIB
    return dict(arena_bytes=extent, memory_max=maximum,
                memory_high=maximum-256*MIB, memlock=extent+128*MIB,
                host_min=maximum+RESERVE+CACHE_LIMIT,
                own_file_cache_limit=CACHE_LIMIT, read_rate=READ_RATE,
                runtime_seconds=600 if tier == 8 else 300, post_seconds=180, quiet_seconds=30,
                max_quiet_wait_seconds=300, full_graph_enabled=False)


def _require(condition, message):
    if not condition:
        raise RuntimeError('Predecessor rejected: ' + message)


def _duration(value, minimum):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _baseline_manifest_digest():
    from candidates.ukl_guarded_loader_v7r3 import protocol as original
    return original.verify_manifest()


def _newest_attempt(tier):
    attempts = []
    pattern = r'tier_%dgib_run_([0-9]{8})_([0-9]{6})_([0-9]+)' % tier
    for path in OUT.glob('tier_%dgib_run_*' % tier):
        match = re.fullmatch(pattern, path.name)
        _require(match is not None and path.is_dir() and not path.is_symlink(),
                 'unexpected predecessor attempt path: ' + str(path))
        attempts.append(((match[1], match[2], int(match[3])), path))
    _require(bool(attempts), 'no %d GiB predecessor attempt exists' % tier)
    # Start timestamp/PID names define attempt order; completion changes mtimes.
    return max(attempts)[1]


def _read_evidence(path):
    _require(not path.is_symlink(), 'symlink evidence: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    _require(len(raw) <= 1024 * 1024, 'oversized evidence: ' + str(path))
    result = json.loads(raw)
    _require(isinstance(result, dict), 'nonobject evidence: ' + str(path))
    return result, hashlib.sha256(raw).hexdigest()


def require_predecessor():
    """Verify the newest previous-tier attempt; never skip a failed attempt.

    These checks read small receipts and source metadata, not graph payload.
    Hash agreement is required within each completed range, never across tiers.
    """
    tier = _tier(TIER)
    current_manifest = verify_manifest()
    current_source = binding()
    previous = 1 if tier == 2 else tier // 2
    path = BASELINE if previous == 1 else _newest_attempt(previous)
    expected_manifest = BASELINE_MANIFEST if previous == 1 else current_manifest
    if previous == 1:
        _require(_baseline_manifest_digest() == BASELINE_MANIFEST,
                 'baseline source manifest changed')
    try:
        evidence, hashes = {}, {}
        for name in ('acceptance.json', 'worker.json', 'before.json'):
            evidence[name], hashes[name] = _read_evidence(path / name)
        accepted, worker, before = (evidence[name] for name in
                                    ('acceptance.json', 'worker.json', 'before.json'))
        expected = budget(previous) if previous != 1 else dict(
            arena_bytes=GIB, memory_max=4*GIB, memory_high=4*GIB-256*MIB,
            memlock=GIB+128*MIB, host_min=GIB+2*GIB+RESERVE+CACHE_LIMIT,
            own_file_cache_limit=CACHE_LIMIT, read_rate=READ_RATE,
            runtime_seconds=300, post_seconds=180, quiet_seconds=30,
            max_quiet_wait_seconds=300, full_graph_enabled=False)
        extent = previous * GIB
        _require(accepted.get('passed') is True and accepted.get('worker_launched') is True
                 and accepted.get('error') is None, 'attempt failed or incomplete')
        _require(accepted.get('manifest_sha256') == expected_manifest
                 and before.get('manifest_sha256') == expected_manifest,
                 'source manifest mismatch')
        _require(accepted.get('worker') == worker, 'embedded and standalone worker receipts differ')
        if previous != 1:
            for name, document in (('acceptance', accepted), ('worker', worker), ('before', before)):
                _require(type(document.get('tier_gib')) is int and document['tier_gib'] == previous,
                         name + ' tier mismatch')
            predecessor = before.get('predecessor')
            _require(isinstance(predecessor, dict) and predecessor.get('passed') is True
                     and predecessor == accepted.get('predecessor')
                     and type(predecessor.get('target_tier_gib')) is int
                     and predecessor['target_tier_gib'] == previous
                     and type(predecessor.get('predecessor_tier_gib')) is int
                     and predecessor['predecessor_tier_gib'] == previous // 2,
                     'saved predecessor chain differs or skips a tier')
        old_source = before['source']
        _require(all(old_source.get(key) == current_source[key] for key in ('path', 'offset', 'identity'))
                 and type(old_source.get('length')) is int and old_source['length'] == extent
                 and worker.get('source_identity') == current_source['identity'],
                 'source identity, path, offset, or extent mismatch')
        _require(all(before['budget'].get(key) == value for key, value in expected.items()),
                 'saved budget mismatch')
        boot = _boot_id()
        qualification = before['monitor_qualification']
        _require(bool(boot) and qualification.get('boot_id') == boot
                 and qualification.get('passed') is True
                 and qualification.get('manifest_sha256') == expected_manifest
                 and qualification.get('scope') == 'monitor_only_no_graph_no_cuda'
                 and qualification.get('cgroup_io_stat_observed') is True
                 and qualification.get('gpu_called') is False
                 and qualification.get('graph_loaded') is False
                 and _duration(qualification.get('duration_seconds'), 180)
                 and type(qualification.get('samples')) is int and qualification['samples'] >= 120,
                 'monitor qualification or current boot mismatch')
        state = accepted['worker_state']
        _require(state.get('LoadState') == 'loaded' and state.get('MainPID') == '0'
                 and state.get('Result') == 'success' and state.get('ExecMainStatus') == '0'
                 and (state.get('ActiveState') == 'inactive'
                      or (state.get('ActiveState') == 'active' and state.get('SubState') == 'exited'))
                 and state.get('MemoryMax') == str(expected['memory_max'])
                 and state.get('MemorySwapMax') == '0'
                 and state.get('LimitMEMLOCK') == str(expected['memlock']),
                 'worker exit state or configured limits invalid')
        _require(all(worker.get(key) is True for key in
                     ('passed', 'direct_io', 'normal_unregister', 'anonymous_memory_released',
                      'vma_absent_after_release', 'limits_verified_before_cuda'))
                 and all(type(worker.get(key)) is int and worker[key] == extent for key in
                         ('extent_bytes', 'loaded_bytes', 'gpu_readback_bytes')),
                 'incomplete GPU readback or release lifecycle')
        digest = worker.get('source_sha256')
        _require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) is not None
                 and digest == worker.get('gpu_sha256'), 'full-range hash mismatch')
        limits = worker['effective_limits']
        for name, wanted in (('memory.max', expected['memory_max']),
                             ('memory.high', expected['memory_high']), ('memory.swap.max', 0),
                             ('pids.max', 64), ('memlock_soft', expected['memlock']),
                             ('memlock_hard', expected['memlock'])):
            _require(type(limits.get(name)) is int and limits[name] == wanted,
                     'effective limit mismatch: ' + name)
        device = '%d:%d' % (os.major(current_source['identity'][0]), os.minor(current_source['identity'][0]))
        _require(isinstance(accepted.get('unit'), str) and bool(accepted['unit'])
                 and limits.get('cgroup') == '/system.slice/' + accepted['unit']
                 and type(limits.get('cpu_quota')) is int and type(limits.get('cpu_period')) is int
                 and 0 < limits['cpu_quota'] <= limits['cpu_period']
                 and limits['io_max'].get(device, {}).get('rbps') == str(READ_RATE),
                 'effective cgroup, CPU, or source read limit mismatch')
        probes = accepted['post_probes']
        _require(isinstance(probes, list) and len(probes) == 2
                 and all(isinstance(probe, dict) and probe.get('passed') is True
                         and type(probe.get('exit_code')) is int and probe['exit_code'] == 0
                         for probe in probes)
                 and {str(Path(probe['path']).parent) for probe in probes}
                     == {str(ROOT / 'results'), '/mnt/n0'}, 'post-run filesystem probes invalid')
        _require(accepted.get('kernel_monitor_ok') is True
                 and accepted.get('post_guard_passed') is True
                 and _duration(accepted.get('post_seconds'), 180)
                 and accepted.get('io_preflight_passed') is True
                 and type(accepted.get('runtime_valid_samples')) is int
                 and accepted['runtime_valid_samples'] >= (11 if previous == 1 else 1)
                 and accepted.get('full_graph_enabled') is False
                 and accepted.get('raw_ssd_access') is False
                 and worker.get('full_graph_enabled') is False,
                 'kernel, post-exit, CPU preflight, or runtime evidence incomplete')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as failure:
        raise RuntimeError('Predecessor evidence missing or malformed: ' + str(path)) from failure
    return dict(passed=True, target_tier_gib=tier, predecessor_tier_gib=previous,
                output=str(path), manifest_sha256=expected_manifest, boot_id=boot,
                source_identity=current_source['identity'], source_sha256=worker['source_sha256'],
                gpu_sha256=worker['gpu_sha256'], evidence_sha256=hashes)


def full_graph_plan():
    """An estimate only; do not use this function as full-graph admission."""
    p = ROOT / 'results/ukl_runtime_prepare_20261001_v6/plan.json'
    old = json.loads(p.read_text())
    audit = json.loads((ROOT/'results/ukl_io_incident_20261001_v1/integrity.json').read_text())
    if not audit['passed'] or not audit['complete'] or len(old['files']) != 8:
        raise RuntimeError('Graph integrity prerequisites missing')
    saved = {f['path']: f for f in audit['files']}
    graph = 0
    for f in old['files']:
        evidence = saved[f['path']]
        st = Path(f['path']).stat()
        if (not evidence['passed'] or st.st_ino != evidence['inode']
                or st.st_size != evidence['bytes'] or st.st_size != f['bytes']
                or evidence['sha256'] != f['sha256']):
            raise RuntimeError('Graph extent/identity differs from integrity evidence')
        graph += (st.st_size+4095)//4096*4096
    nodes = 787801471
    hot = (nodes//10) * 512
    slots = nodes * 4
    window = 320 * 1024 * 8
    runtime_allowance = 32 * GIB
    peak = graph + hot + slots + window + runtime_allowance + CACHE_LIMIT
    return dict(graph_bytes=graph, cpu_hot_features_bytes=hot,
                cpu_slot_map_bytes=slots, root_window_bytes=window,
                proposed_runtime_allowance_bytes=runtime_allowance,
                runtime_allowance_measured=False, own_file_cache_budget=CACHE_LIMIT,
                estimated_peak_bytes=peak, host_reserve_bytes=RESERVE,
                proposed_host_requirement_bytes=peak+RESERVE,
                full_graph_enabled=False,
                warning='Planning estimate, not runtime admission. Sampler/model/driver/IO '
                        'peaks and unaligned graph tails remain unvalidated.')


def source_device():
    st = SOURCE.stat()
    device = Path('/dev/block') / ('%d:%d' % (os.major(st.st_dev), os.minor(st.st_dev)))
    resolved = device.resolve(strict=True)
    if not resolved.is_block_device():
        raise RuntimeError('Source is not on a recognized block device')
    return str(resolved), '%d:%d' % (os.major(st.st_dev), os.minor(st.st_dev))
