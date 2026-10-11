"""Metadata-only budgets; only a 1 GiB real graph extent is executable."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ukl_guarded_loader_20261002_v7/repair3'
GIB = 1024**3
MIB = 1024**2
EXTENT = GIB
SOURCE = Path('/mnt/n0/digit/ukl_sparse_overlay_v3/full_csc_v1/indices.i32')
RESERVE = 64 * GIB
HEADROOM = 2 * GIB
CACHE_LIMIT = 128 * MIB
READ_RATE = 128 * MIB
LOCK = ROOT / 'results/ukl_runtime_prepare_20261001_v6/smoke.lock'
PYTHON = '/home/embed/miniconda3/envs/gids/bin/python'


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
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
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
    if d['path'] != str(SOURCE) or d['offset'] != 0 or d['length'] != EXTENT:
        raise RuntimeError('Only the fixed first 1 GiB graph extent is admitted')
    if SOURCE.is_symlink() or identity(SOURCE.stat()) != d['identity']:
        raise RuntimeError('Graph source identity changed')
    return d


def budget():
    # Direct I/O avoids charging a second graph-sized page cache to this worker.
    return dict(arena_bytes=EXTENT, memory_max=4*GIB,
                memory_high=4*GIB-256*MIB, memlock=EXTENT+128*MIB,
                host_min=EXTENT+HEADROOM+RESERVE+CACHE_LIMIT,
                own_file_cache_limit=CACHE_LIMIT, read_rate=READ_RATE,
                runtime_seconds=300, post_seconds=180, quiet_seconds=30,
                max_quiet_wait_seconds=300, full_graph_enabled=False)


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
