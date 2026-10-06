"""Step 1/2: owned anonymous UKL native correctness, without features or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

from . import protocol as P
from . import sampling as S
from . import checks as K
from . import fork_guard as H
from candidates.ukl_real_multi_v9.memory import Arena, Registration, RETAINED, identity
from candidates.ukl_real_multi_v9.worker import unmapped

FRAGMENTS = P.ROOT / 'results/ukl_real_fragment_20261001_v1/fragments.json'
ARRAY_NAMES = dict(ptr='indptr', idx='indices', gptr='group_ptr', gidx='group_ids',
                   covered='covered', bases='bases', primary='primary', members='members')
FANOUTS, SEEDS = (1, 5, 10, 32), (0, 23)


def write(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if len(raw.encode()) > P.MIB:
        raise RuntimeError('Worker JSON exceeds 1 MiB bound')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(raw)
    os.replace(str(temporary), str(path))


def limits():
    expected = P.budget()
    cgroup = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if line.startswith('0::'))
    folder = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
    values = {key: int((folder / key).read_text()) for key in
              ('memory.max', 'memory.high', 'memory.swap.max', 'pids.max')}
    quota, period = map(int, (folder / 'cpu.max').read_text().split())
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    devices = dict((line.split()[0], dict(item.split('=') for item in line.split()[1:]))
                   for line in (folder / 'io.max').read_text().splitlines())
    values.update(cgroup=cgroup, cpu_quota=quota, cpu_period=period,
                  memlock_soft=soft, memlock_hard=hard, io_max=devices)
    _, device = P.source_device()
    if not P._limits_valid(values, expected, device):
        raise RuntimeError('Effective cgroup/MEMLOCK/I/O limits differ from the v10r4 plan')
    return values


def admission(remaining, stopped):
    stopped()
    if type(remaining) is not int or not 0 <= remaining <= 256 * P.GIB:
        raise RuntimeError('Remaining allocation outside the admitted bound')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
            break
    required = remaining + P.HEADROOM + P.RESERVE
    if available is None or available < required:
        raise RuntimeError('Host memory admission refused: available=%s required=%s' %
                           (available, required))


def progress_callback(event, check, rate):
    started, last = time.monotonic(), [0]

    def progress(name, done, length, total_done, total):
        check()
        while True:
            delay = total_done / rate - (time.monotonic() - started)
            if delay <= 0:
                break
            time.sleep(min(delay, 0.2))
            check()
        if total_done - last[0] >= 256 * P.MIB or done == length:
            event('load_progress', array=name, array_bytes_done=done, array_bytes=length,
                  allocated_bytes_done=total_done, allocated_bytes=total,
                  remaining_bytes=total - total_done)
            last[0] = total_done
    return progress


def array_receipts(arena):
    result = {}
    for name, spec in arena.specs.items():
        logical, allocated = arena.lengths[name], arena.allocated_lengths[name]
        start = arena.offsets[name]
        K.require(not any(arena.mm[start + logical:start + allocated]), 'Nonzero tail padding: ' + name)
        K.require(arena.digests[name] == spec['sha256'], 'Whole-file digest differs: ' + name)
        result[name] = dict(path=str(spec['path']), offset=spec['offset'], logical_length=logical,
                            allocated_bytes=allocated, arena_offset=start,
                            padding_bytes=allocated - logical, cpu_sha256=arena.digests[name],
                            expected_sha256=spec['sha256'], expected_sha256_matched=True,
                            cpu_padding_zero=True, source_identity=list(arena.source_identities[name]))
    return result


def revalidate(sources):
    K.require(P.binding() == sources, 'Graph source binding changed during worker lifecycle')


def close_owned(arena, registration, graph, cpu, gpu):
    """The first failed synchronization stops release and preserves ownership."""
    if gpu is not None and gpu.retained:
        raise RuntimeError('Native GPU operation retained ownership; release refused')
    if gpu is not None:
        gpu.close()
    if cpu is not None:
        cpu.close()
    if graph is not None:
        graph.close()
    if registration is not None:
        registration.close()
    if arena is not None:
        arena.close()


def fixture_payload(high=False):
    """Tiny independently generated CSC with empty, duplicate, and grouped rows."""
    rng, nodes = np.random.default_rng(20261002), 67
    origin = 2**33 + 17 if high else 0
    ptr, idx, gptr, members, covered = [origin], [], [0], [], []
    for owner in range(nodes):
        row = [] if owner == 1 else rng.integers(0, nodes, size=2049 if owner == 0 else 19).tolist()
        begin = ptr[-1]
        for at in range(0, len(row) - 1, 3):
            if row[at] != row[at + 1]:
                members.append(row[at:at + 2])
                covered.extend((begin + at, begin + at + 1))
        idx.extend(row)
        ptr.append(origin + len(idx))
        gptr.append(len(members))
    groups = len(members)
    arrays = dict(ptr=np.asarray(ptr, np.int64), idx=np.asarray(idx, np.int32),
                  gptr=np.asarray(gptr, np.int64), gidx=np.arange(groups, dtype=np.int32),
                  members=np.asarray(members, np.int32), covered=np.asarray(covered, np.int64),
                  bases=np.arange(groups, dtype=np.int64) * 2 + 2**32 + 128,
                  primary=np.arange(nodes, dtype=np.int64) + 2**32)
    metadata = dict(nodes=nodes, edges=len(idx), groups=groups, units=groups,
                    storage_rows=2**32 + 128 + groups * 2, origin=origin, unit_origin=0)
    return arrays, metadata


def fixture_specs(folder, arrays):
    folder.mkdir()
    total, specs = 0, {}
    for key, value in arrays.items():
        path, payload = folder / (key + '.bin'), value.tobytes()
        total += len(payload)
        K.require(total <= P.MIB, 'Synthetic payload exceeds 1 MiB')
        with path.open('xb') as stream:
            stream.write(payload)
        specs[ARRAY_NAMES[key]] = dict(path=path, offset=0, length=len(payload),
                                      identity=identity(path.stat()),
                                      sha256=hashlib.sha256(payload).hexdigest())
    return specs


def raw_matrix(graph, cpu, gpu, roots, check):
    rows, high = [], False
    for arm in P.ARMS:
        packed = grouped = arm == 'digit'
        for fanout in FANOUTS:
            for seed in SEEDS:
                check()
                left = cpu.sample(roots, fanout, grouped, seed, packed)
                right = gpu.sample(roots, fanout, grouped, seed, packed)
                K.equal_arrays(left, right, 'CPU/GPU raw output')
                K.raw_checks(graph, roots, fanout, left, grouped, packed)
                compact = K.raw_checks(graph, roots, fanout, right, grouped, packed)
                high = high or bool(np.any(compact['eid'] > 2**32 - 1))
                rows.append(dict(arm=arm, fanout=fanout, seed=seed,
                                 edges=len(compact['eid']), raw_sha256=K.raw_digest(right),
                                 cpu_raw_sha256=K.raw_digest(left)))
    return rows, high


def synthetic(out, check, admitted, event, ownership):
    receipts, high, peak = [], False, 0
    for shifted in (False, True):
        arena = registration = graph = cpu = gpu = None
        release_attempted = False
        try:
            payload, metadata = fixture_payload(shifted)
            specs = fixture_specs(out / ('fixture_high' if shifted else 'fixture_normal'), payload)
            del payload
            arena = Arena(specs, admission_check=admitted, max_bytes=P.MIB)
            ownership.track(arena)
            dontfork = H.dontfork_arena(arena)
            peak = max(peak, arena.size)
            arena.load(lambda *_: check())
            address, size = arena.address, arena.size
            graph = S.Graph(arena, metadata)
            cpu = S.Native(graph)
            before_register = H.assert_dontfork(arena)
            registration = Registration(arena)
            after_register = H.assert_dontfork(arena)
            gpu = S.Native(graph, registration)
            rows, saw_high = raw_matrix(graph, cpu, gpu,
                                        np.asarray([0, 1, 2, 5, 16, 66], np.int32), check)
            high = high or saw_high
            release_attempted = True
            close_owned(arena, registration, graph, cpu, gpu)
            ownership.confirm_released(arena)
            arena = registration = graph = cpu = gpu = None
            K.require(unmapped(address, size) and not RETAINED and not S.RETAINED,
                      'Synthetic anonymous VMA or registration retained')
            receipts.append(dict(high_origin=shifted, draws=rows, vma_absent=True,
                                 arena_bytes=size, dontfork_before_load=dontfork,
                                 dontfork_before_registration=before_register,
                                 dontfork_after_registration=after_register))
            event('synthetic_case_complete', high_origin=shifted, comparisons=len(rows))
        finally:
            if not release_attempted:
                close_owned(arena, registration, graph, cpu, gpu)
                if arena is not None:
                    ownership.confirm_released(arena)
    K.require(high, 'Synthetic high EID was not exercised')
    return dict(passed=True, cases=2, draws=32, high_eid_exercised=True,
                baseline_packed=False, digit_packed=True, normal_release=True,
                peak_arena_bytes=peak, receipts=receipts)


def full_roots(graph):
    with FRAGMENTS.open('rb') as stream:
        raw = stream.read(P.MIB + 1)
    K.require(len(raw) <= P.MIB, 'Fragment evidence exceeds bound')
    evidence = json.loads(raw)
    K.require(evidence.get('passed') is True and len(evidence['fragments']) == 5,
              'Frozen real owner evidence invalid')
    owners = [0, 1, graph.nodes - 1]
    high = False
    for row in evidence['fragments']:
        owner = row['owner']
        K.require(type(owner) is int and 0 <= owner < graph.nodes, 'Fragment owner outside graph')
        begin, end = map(int, graph.arrays['ptr'][owner:owner + 2])
        K.require(begin == row['eid_start'] and end - begin == row['degree'],
                  'Frozen owner range differs from loaded full graph')
        high = high or (begin > 2**32 - 1 and end > begin)
        owners.append(owner)
    K.require(high, 'No verified real high-EID owner')
    return np.asarray(sorted(set(owners)), dtype=np.int32)


def batch_roots(nodes, batch):
    K.require(nodes >= P.ROOTS, 'Insufficient nodes for bounded roots')
    rng = np.random.default_rng(np.random.SeedSequence([20261002, batch]))
    # Keep only <=2048 candidates at a time; never choice/permutation over N.
    chosen, seen = [], set()
    while len(chosen) < P.ROOTS:
        for value in rng.integers(0, nodes, size=2 * (P.ROOTS - len(chosen)), dtype=np.int32):
            value = int(value)
            if value not in seen:
                chosen.append(value)
                seen.add(value)
                if len(chosen) == P.ROOTS:
                    break
    return np.asarray(chosen, np.int32)


def one_batch(graph, cpu, gpu, roots, arm, batch, check):
    packed, captured = arm == 'digit', {'cpu': {}, 'gpu': {}}

    def audit(which):
        def callback(layer, seeds, fanout, grouped, actual_packed, raw):
            check()
            K.require(actual_packed == packed and grouped == (packed and layer == 0),
                      'Arm packed/grouped layer behavior changed')
            K.raw_checks(graph, seeds, fanout, raw, grouped, packed)
            captured[which][layer] = raw
        return callback

    samplers = [S.Sampler(native, grouped=packed, seed=0) for native in (cpu, gpu)]
    left = samplers[0].layers(roots, batch=batch, audit=audit('cpu'))
    right = samplers[1].layers(roots, batch=batch, audit=audit('gpu'))
    K.require(np.array_equal(left[0], right[0]), 'CPU/GPU input frontier differs')
    for index, ((ldst, lsrc, lraw), (rdst, rsrc, rraw)) in enumerate(zip(left[1], right[1])):
        K.require(np.array_equal(ldst, rdst) and np.array_equal(lsrc, rsrc), 'CPU/GPU layer frontier differs')
        K.equal_arrays(lraw, rraw, 'CPU/GPU compact output')
        K.equal_arrays(captured['cpu'][index], captured['gpu'][index], 'CPU/GPU all raw outputs')
    check()
    cpu_blocks = samplers[0].sample_blocks(roots, batch=batch, layers=left)
    gpu_blocks = samplers[1].sample_blocks(roots, batch=batch, layers=right)
    cpu_receipts = K.block_checks(graph, left[1], cpu_blocks, roots, packed)
    gpu_receipts = K.block_checks(graph, right[1], gpu_blocks, roots, packed)
    K.require(cpu_receipts == gpu_receipts, 'CPU/GPU block or storage identity differs')
    layers = []
    for index, ((dst, src, raw), block) in enumerate(zip(right[1], gpu_receipts)):
        hashes = dict(raw_sha256=K.raw_digest(captured['gpu'][index]),
                      frontier_sha256=K.digest(src), eid_sha256=K.digest(raw['eid']),
                      row_sha256=K.digest(raw['rows']), **block)
        ldst, lsrc, lraw = left[1][index]
        cpu_hashes = dict(raw_sha256=K.raw_digest(captured['cpu'][index]),
                          frontier_sha256=K.digest(lsrc), eid_sha256=K.digest(lraw['eid']),
                          row_sha256=K.digest(lraw['rows']), **cpu_receipts[index])
        K.require(hashes == cpu_hashes, 'CPU/GPU layer receipt hash differs')
        layers.append(dict(layer=index, fanout=P.FANOUTS[index], root_count=len(dst),
                           frontier_count=len(src), edge_count=len(raw['eid']), **hashes,
                           **{'cpu_' + key: value for key, value in cpu_hashes.items()}))
    return dict(batch=batch, root_count=len(roots), roots_sha256=K.digest(roots),
                layer_count=3, fanouts=list(P.FANOUTS), layers=layers,
                input_nodes_count=len(right[0]), input_nodes_sha256=K.digest(right[0]),
                cpu_gpu_raw_parity=True, frontier_parity=True, eid_parity=True,
                block_parity=True, address_checks=True, storage_mapping=True)


def full_sampling(graph, cpu, gpu, check, event):
    roots = full_roots(graph)
    rows, high = raw_matrix(graph, cpu, gpu, roots, check)
    K.require(high, 'Full graph high EID sampling not exercised')
    single = dict(passed=True, cases=len(rows), fanouts=list(FANOUTS), seeds=list(SEEDS),
                  high_eid_exercised=high, raw_parity=True, address_checks=True,
                  root_count=len(roots), roots_sha256=K.digest(roots), receipts=rows)
    event('single_layer_complete', comparisons=len(rows), high_eid_exercised=high)
    batches, arms = [batch_roots(graph.nodes, batch) for batch in range(P.BATCHES)], {}
    for arm in P.ARMS:
        arms[arm] = []
        for batch, roots in enumerate(batches):
            check()
            record = one_batch(graph, cpu, gpu, roots, arm, batch, check)
            arms[arm].append(record)
            event('batch_complete', arm=arm, batch=batch,
                  input_nodes=record['input_nodes_count'],
                  sampled_edges=sum(row['edge_count'] for row in record['layers']))
    return dict(passed=True, single_layer=single, arms=arms)


def execute(out):
    P.verify_manifest()
    sources, metadata = P.binding(), P.metadata()
    K.require(len(sources) == 8 and all(spec.get('sha256') for spec in sources.values()),
              'Eight frozen whole-file hashes required')
    actual_limits = limits()
    # Hash both binaries before either CDLL initialization or allocation.
    S._binary(False)
    S._binary(True)
    write(out / 'effective_limits.json', actual_limits)
    events, terminate = [], [False]
    arena = registration = graph = cpu = gpu = ownership = None
    release_attempted = False

    def event(stage, **kwargs):
        events.append(dict(time=time.time(), stage=stage, selected_stage=P.STAGE, **kwargs))
        write(out / 'lifecycle.json', events)

    def stopped():
        if terminate[0] or (out / 'STOP').exists():
            raise RuntimeError('Controller requested a cooperative stop')

    def admitted(remaining):
        admission(remaining, stopped)

    def check():
        admitted(arena.remaining_bytes if arena is not None else 0)

    def request_stop(signum, frame):
        terminate[0] = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    try:
        check()
        # Torch's first import calls platform.uname(), which may fork uname -p.
        # Complete all framework imports and a real CPU block before ANY CUDA
        # registration or graph arena exists, then forbid child processes.
        prewarm = H.prewarm_cpu_block()
        event('cpu_framework_prewarm_complete', receipt=prewarm)
        ownership = H.OwnershipGuard().enter()
        event('nofork_guard_entered', receipt=ownership.receipt())
        fixtures = synthetic(out, check, admitted, event, ownership)
        event('synthetic_complete', cases=fixtures['cases'], comparisons=fixtures['draws'])
        check()
        revalidate(sources)
        event('load', arena_bytes=P.EXTENT, arrays=list(sources), buffered_fallback=False,
              application_read_rate=P.APPLICATION_READ_RATE, hard_read_rate=P.READ_RATE)
        arena = Arena(sources, admission_check=admitted, max_bytes=P.EXTENT)
        ownership.track(arena)
        dontfork = H.dontfork_arena(arena)
        K.require(arena.size == P.EXTENT, 'Anonymous allocation differs from full graph plan')
        arena.load(progress_callback(event, check, P.APPLICATION_READ_RATE))
        check()
        logical = sum(arena.lengths.values())
        K.require(arena.loaded_bytes == arena.size and arena.loaded_payload_bytes == logical,
                  'Incomplete physical/logical array loading')
        arrays = array_receipts(arena)
        revalidate(sources)
        event('loaded', loaded_bytes=arena.loaded_bytes, loaded_payload_bytes=logical,
              digests=arena.digests, direct_io=True, source_revalidated=True)
        address, size = arena.address, arena.size
        graph = S.Graph(arena, metadata)
        cpu = S.Native(graph)
        event('register')
        before_register = H.assert_dontfork(arena)
        registration = Registration(arena)
        after_register = H.assert_dontfork(arena)
        gpu = S.Native(graph, registration)
        event('registered')
        sampling = full_sampling(graph, cpu, gpu, check, event)
        sampling['synthetic'] = fixtures
        check()
        event('release')
        release_attempted = True
        close_owned(arena, registration, graph, cpu, gpu)
        ownership.confirm_released(arena)
        arena = registration = graph = cpu = gpu = None
        K.require(not RETAINED and not S.RETAINED and unmapped(address, size),
                  'Anonymous allocation, native scratch, or registration not released')
        guarded = ownership.receipt()
        ownership.release()
        released_guard = ownership.receipt()
        revalidate(sources)
        event('complete', vma_absent=True, source_revalidated=True)
        report = dict(passed=True, application_read_rate=P.APPLICATION_READ_RATE, hard_read_rate=P.READ_RATE, stage=P.STAGE, arena_bytes=size, logical_bytes=logical,
                      loaded_bytes=size, loaded_payload_bytes=logical, padding_bytes=size - logical,
                      arrays=arrays, metadata=metadata, sampling=sampling, direct_io=True,
                      normal_unregister=True, anonymous_memory_released=True,
                      cpu_framework_prewarmed_before_any_registration=True,
                      cpu_framework_prewarm=prewarm,
                      nofork_guard_before_any_arena=True,
                      nofork_guard_held_through_release=True,
                      nofork_guard_released_after_all_arenas=True,
                      nofork_guard_receipt=guarded,
                      nofork_guard_release_receipt=released_guard,
                      all_arenas_dontfork_verified=True,
                      full_arena_dontfork_before_load=dontfork,
                      full_arena_dontfork_before_registration=before_register,
                      full_arena_dontfork_after_registration=after_register,
                      vma_absent_after_release=True, limits_verified_before_cuda=True,
                      effective_limits=actual_limits, source_revalidated_before_cuda=True,
                      source_revalidated_after_release=True,
                      maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      full_graph_enabled=True, full_graph_load=True, graph_sampling_enabled=True,
                      model_enabled=False, features_enabled=False, raw_ssd_access=False,
                      full_gpu_readback_performed=False, timed_performance_enabled=False,
                      scope='Full UKL owned-anonymous native sampling correctness and bounded DGL blocks only')
        K.require(P.validate_worker_report(report), 'Worker report fails the frozen receipt contract')
        write(out / 'worker.json', report)
    finally:
        try:
            if not release_attempted:
                close_owned(arena, registration, graph, cpu, gpu)
                if arena is not None and ownership is not None:
                    ownership.confirm_released(arena)
                if ownership is not None and ownership.active:
                    # release() refuses to disable the barrier if any tracked
                    # arena is live, registered, or still mapped.
                    ownership.release()
            # If release_attempted is true and close_owned failed, do not
            # retry release or disable the guard; preserve the original CUDA
            # error and let its retained owners keep the barrier active.
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('sampling',), required=True)
    args = parser.parse_args()
    if os.environ.get('UKL_V10R4_BOUNDED_WORKER') != '1':
        raise RuntimeError('Use the bounded v10r4 controller')
    P.configure_stage(args.stage)
    execute(Path(os.environ['UKL_V10R4_OUTPUT']).resolve(strict=True))


if __name__ == '__main__':
    main()
