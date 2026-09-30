"""Phase 9D bounded preprocessing primitives, not a complete artifact builder.

Can run by filename to avoid importing digit/__init__ (and Torch/DGL). Inputs
are regular NPY files (C/F-order edges, C-order features). Scratch is retained.
"""

# Locate DiGiT independently of the checkout directory and working directory.
from pathlib import Path as _DigitPath
import sys as _digit_sys
_digit_root = next((p for p in _DigitPath(__file__).resolve().parents
                    if (p / ".digit-root").is_file()), None)
if _digit_root is None:
    raise RuntimeError("Cannot locate the DiGiT project root")
if str(_digit_root) not in _digit_sys.path:
    _digit_sys.path.insert(0, str(_digit_root))
import digit_paths as _digit_paths

import argparse
import fcntl
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import struct
import time

import numpy as np

MIB = 1024 ** 2
RECORD = np.dtype([("dst", "<i8"), ("eid", "<i8"), ("src", "<i8")])
SCHEMA = "digit-large-preprocess-v1"
STOP = False


def header(path):
    path = Path(path).resolve(strict=True)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("input must be a regular file: " + str(path))
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError("only NPY v1/v2 headers are supported")
        shape, fortran, dtype = reader(f)
        offset = f.tell()
    if dtype.hasobject or dtype.fields or dtype.kind not in "iufb":
        raise ValueError("require plain numeric NPY input")
    size = math.prod(shape) * dtype.itemsize
    if path.stat().st_size != offset + size:
        raise ValueError("NPY payload length mismatch: " + str(path))
    return dict(path=str(path), shape=list(shape), dtype=dtype.str, fortran_order=fortran,
                offset=offset, payload_bytes=size, file_bytes=offset + size)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(MIB), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    info = header(path)
    before = Path(info["path"]).stat()
    info["sha256"] = digest(info["path"])
    after = Path(info["path"]).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("input changed during hashing")
    info["mtime_ns"] = after.st_mtime_ns
    return info


def inputs_unchanged(binding):
    for info in binding.values():
        if isinstance(info, dict) and "mtime_ns" in info:
            current = Path(info["path"]).stat()
            if current.st_size != info["file_bytes"] or current.st_mtime_ns != info["mtime_ns"]:
                raise ValueError("input changed while job was running; cannot publish completion")


def edge_count(info):
    shape = info["shape"]
    if len(shape) != 2 or 2 not in shape or np.dtype(info["dtype"]).kind not in "iu":
        raise ValueError("edges must be integer [2,E] or [E,2]")
    return shape[1] if shape[0] == 2 else shape[0]


def validate_features(info):
    if len(info["shape"]) != 2 or min(info["shape"]) <= 0 or info["fortran_order"]:
        raise ValueError("features must be nonempty C-order [N,D]")


def nearest_existing(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def free_bytes(path):
    s = os.statvfs(nearest_existing(path))
    return s.f_bavail * s.f_frsize


def memory_snapshot():
    """Read host MemAvailable and visible cgroup-v2 ancestor headroom."""
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    limits = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            root = Path("/sys/fs/cgroup")
            path = root / line[3:].lstrip("/")
            while path == root or root in path.parents:
                maximum, current = path / "memory.max", path / "memory.current"
                if maximum.is_file() and current.is_file():
                    text = maximum.read_text().strip()
                    if text != "max":
                        limits.append(max(0, int(text) - int(current.read_text())))
                if path == root:
                    break
                path = path.parent
    candidates = limits + ([] if available is None else [available])
    return dict(host_available_bytes=available, visible_cgroup_v2_headroom_bytes=limits,
                effective_available_bytes=min(candidates) if candidates else None,
                note="Read-only snapshot, not a reservation; cgroup-v1 is not inferred")


def workspace_estimate(n, e, row_bytes, group_size, slot_bytes, ratio,
                       chunk_edges, fan_in):
    if n <= 0 or e < 0 or row_bytes <= 0 or group_size <= 0:
        raise ValueError("invalid graph/feature/group sizes")
    if not math.isfinite(ratio) or ratio < 0 or chunk_edges <= 0 or fan_in < 2:
        raise ValueError("invalid replication/chunk/fan-in")
    if slot_bytes < group_size * row_bytes or slot_bytes % row_bytes:
        raise ValueError("slot must fit the group and contain whole rows")
    slots = slot_bytes // row_bytes
    primary = n // group_size
    replica = int(math.floor(ratio * n)) // group_size
    groups = primary + replica
    # Maximise n - g*primary_used + slots*(primary_used+replicas).
    rows = n + primary * (slots - group_size) + replica * slots + 2 * (slots - 1)
    runs = max(1, (e + chunk_edges - 1) // chunk_edges)
    count, passes, files = runs, 0, runs
    while count > 1:
        count = (count + fan_in - 1) // fan_in
        files += count
        passes += 1
    csc = 8 * (e + n + 1)
    # Retain runs from every merge level, headers + manifest/hash metadata.
    sort_scratch = 24 * e * (passes + 1) + csc + 4096 * (files + 4)
    # Upper bound final arrays; original CSC is scratch, not bundle content.
    arrays = (8 * groups * group_size + 24 * groups + 8 * rows + 8 * n
              + 8 * (n + groups + 1) + 8 * e + 9 * 4096)
    feature_payload = rows * row_bytes
    # Feature shards plus assembled NPY coexist for resumability.
    feature_scratch = feature_payload + 4096 * ((rows + 16383) // 16384 + 2)
    return dict(num_nodes=n, num_edges=e, feature_row_bytes=row_bytes,
                max_primary_groups=primary, max_replica_groups=replica,
                storage_rows_upper_bound=rows, feature_payload_upper_bound=feature_payload,
                final_artifact_upper_bound=arrays + feature_payload,
                external_sort_retained_scratch_upper_bound=sort_scratch,
                feature_shard_scratch_upper_bound=feature_scratch,
                known_disk_reservation_bytes=sort_scratch + arrays + feature_payload + feature_scratch,
                sort_initial_runs=runs, merge_passes=passes,
                sort_checkpoint_files=files + 2,
                legacy_csc_int64_array_scale_estimate=32 * e + 16 * n,
                legacy_grouping_edge_arrays_lower_bound=10 * e,
                legacy_estimate_is_peak=False,
                complete_grouping_scratch_budget_known=False)


def capacity_plan(edges, features, output, group_size=4, slot_bytes=8192,
                  ratio=0.2, chunk_edges=262144, fan_in=16, memory_mib=256):
    ef, ff = header(edges), header(features)
    validate_features(ff)
    estimate = workspace_estimate(ff["shape"][0], edge_count(ef),
                                  ff["shape"][1] * np.dtype(ff["dtype"]).itemsize,
                                  group_size, slot_bytes, ratio, chunk_edges, fan_in)
    reserve = estimate["known_disk_reservation_bytes"]
    available = free_bytes(output)
    heap = sort_heap_bound(chunk_edges, fan_in, estimate["sort_checkpoint_files"])
    return dict(schema=SCHEMA, mode="header_only_read_only_plan", edges=ef, features=ff,
                output=str(Path(output).resolve()), estimate=estimate,
                memory=dict(buffer_budget_bytes=memory_mib * MIB,
                            sort_explicit_heap_bound_bytes=heap,
                            buffer_budget_passed=heap <= memory_mib * MIB,
                            system=memory_snapshot(),
                            note="Not total RSS: interpreter/libraries need separate address-space limit"),
                disk=dict(available_bytes=available, known_reservation_bytes=reserve,
                          with_20_percent_headroom=math.ceil(reserve * 1.2),
                          known_stages_fit=math.ceil(reserve * 1.2) <= available),
                full_artifact_build_approved=False, input_payload_scanned=False,
                blockers=["Group/replica/owner-order/rewritten-CSC externalization pending",
                          "No paper-scale source content validation or measured RSS gate yet"],
                raw_ssd_accessed=False)


def sort_heap_bound(chunk_edges, fan_in, files):
    return 192 * chunk_edges + (fan_in + 2) * 65536 + files * 4096 + 4 * MIB


def check_heap(bound, memory_mib):
    if bound > memory_mib * MIB:
        raise ValueError("buffer/manifest budget exceeded; reduce chunk/batch size")
    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    vm = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    if soft != resource.RLIM_INFINITY and vm + bound > soft:
        raise ValueError("planned buffers do not fit the address-space ceiling")
    available = memory_snapshot()["effective_available_bytes"]
    if available is not None and bound > available * .8:
        raise ValueError("planned buffers exceed 80% of current available memory")


def check_space(path, size):
    if math.ceil(size * 1.2) > free_bytes(path):
        raise ValueError("insufficient free space including 20% headroom")


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fsync_dir(path.parent)


class Paused(Exception):
    pass


class Job:
    """Immutable task outputs + atomic hash-bound completion manifest.

    Only uncommitted .part files and unreferenced expected outputs may be
    replaced. Corrupt committed files fail closed; no silent restart.
    """

    def __init__(self, root, binding, stop_after=None):
        self.root = Path(root).resolve()
        self.binding = binding
        self.stop_after = stop_after
        self.committed = 0

    def __enter__(self):
        if self.root.exists() and not (self.root / "state.json").exists():
            if any(p.name != ".lock" for p in self.root.iterdir()):
                raise ValueError("new workspace must be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / "state.json"
            if path.exists():
                self.state = _digit_paths.json_loads(path.read_text())
                if self.state["binding"] != self.binding or self.state["schema"] != SCHEMA:
                    raise ValueError("checkpoint binding changed; use a NEW workspace")
                for name, info in self.state["files"].items():
                    if Path(name).name != name:
                        raise ValueError("invalid checkpoint filename")
                    file = self.root / name
                    if file.stat().st_size != info["bytes"] or digest(file) != info["sha256"]:
                        raise ValueError("committed checkpoint corrupt: " + name)
            else:
                if any(p.name != ".lock" for p in self.root.iterdir()):
                    raise ValueError("new workspace must be empty")
                self.state = dict(schema=SCHEMA, binding=self.binding, files={}, phase="running")
                self.save()
            return self
        except BaseException:
            self.lock.close()
            raise

    def save(self):
        self.state["updated_unix"] = time.time()
        self.state["pid"] = os.getpid()
        self.state["completed_files"] = len(self.state["files"])
        # Linux ru_maxrss may include the forked test parent's pre-exec peak.
        # VmHWM describes this executable's address space instead.
        status = dict((line.split(":", 1)[0], line.split(":", 1)[1].strip())
                      for line in Path("/proc/self/status").read_text().splitlines())
        self.state["peak_rss_bytes"] = int(status["VmHWM"].split()[0]) * 1024
        self.state["current_rss_bytes"] = int(status["VmRSS"].split()[0]) * 1024
        self.state["rusage_maxrss_including_preexec_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        self.state["address_space_limit_bytes"] = resource.getrlimit(resource.RLIMIT_AS)[0]
        atomic_json(self.root / "state.json", self.state)

    def __exit__(self, typ, value, tb):
        try:
            if typ is not None:
                self.state["phase"] = "paused" if typ is Paused else "failed"
                self.save()
        finally:
            self.lock.close()

    def task(self, name, writer):
        if name in self.state["files"]:
            return self.root / name
        if STOP or (self.stop_after is not None and self.committed >= self.stop_after):
            raise Paused()
        self.state["phase"] = "running"
        self.state["active_task"] = name
        self.save()
        temp = self.root / (name + ".part")
        output = self.root / name
        if output.exists():
            # Crash after rename but before manifest commit: reuse the orphan
            # inode as scratch instead of temporarily keeping two full outputs.
            os.replace(output, temp)
            fsync_dir(self.root)
        with temp.open("wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, output)
        fsync_dir(self.root)
        self.state["files"][name] = dict(bytes=output.stat().st_size, sha256=digest(output))
        self.committed += 1
        self.save()
        print(json.dumps(dict(task=name, committed_files=len(self.state["files"]),
                              bytes=output.stat().st_size)), flush=True)
        return output

    def complete(self, outputs):
        inputs_unchanged(self.binding)
        self.state.update(phase="complete", active_task=None, outputs=outputs,
                          raw_ssd_accessed=False, complete_artifact=False)
        self.save()
        return self.state


def read_items(f, info, start, count):
    dtype = np.dtype(info["dtype"])
    f.seek(info["offset"] + start * dtype.itemsize)
    data = f.read(count * dtype.itemsize)
    if len(data) != count * dtype.itemsize:
        raise ValueError("short input read")
    return np.frombuffer(data, dtype=dtype)


def npy_header(f, shape, dtype):
    np.lib.format.write_array_header_1_0(
        f, dict(descr=np.dtype(dtype).str, fortran_order=False, shape=tuple(shape)))


def records(path):
    with Path(path).open("rb") as f:
        while True:
            block = f.read(24 * 2048)
            if not block:
                break
            if len(block) % 24:
                raise ValueError("truncated sort run")
            for item in struct.iter_unpack("<qqq", block):
                yield item


def merge(paths, out):
    streams = [records(path) for path in paths]
    buffer = bytearray()
    try:
        for item in heapq.merge(*streams):
            buffer.extend(struct.pack("<qqq", *item))
            if len(buffer) >= 65536:
                out.write(buffer)
                buffer.clear()
        out.write(buffer)
    finally:
        for stream in streams:
            stream.close()


def build_csc(edges, features, workspace, chunk_edges=262144, fan_in=16,
              memory_mib=256, stop_after=None):
    # Header preflight before expensive hashes or workspace creation.
    ef, ff = header(edges), header(features)
    validate_features(ff)
    n, e = ff["shape"][0], edge_count(ef)
    estimate = workspace_estimate(n, e, 1, 1, 1, 0, chunk_edges, fan_in)
    check_heap(sort_heap_bound(chunk_edges, fan_in, estimate["sort_checkpoint_files"]), memory_mib)
    check_space(workspace, estimate["external_sort_retained_scratch_upper_bound"])
    binding = dict(kind="stable_csc", edges=identity(edges), features=identity(features),
                   chunk_edges=chunk_edges, fan_in=fan_in, memory_mib=memory_mib,
                   code_sha256=digest(__file__), numpy=np.__version__)
    if any(binding["edges"][k] != v for k, v in ef.items()) or any(
            binding["features"][k] != v for k, v in ff.items()):
        raise ValueError("input header changed during preflight")
    with Job(workspace, binding, stop_after) as job:
        paths = []
        with Path(ef["path"]).open("rb") as source:
            for index, start in enumerate(range(0, max(e, 1), chunk_edges)):
                count = min(chunk_edges, e - start)
                def write_run(out, start=start, count=count):
                    components_contiguous = ((ef["shape"][0] == 2) != ef["fortran_order"])
                    if components_contiguous:
                        src = read_items(source, ef, start, count)
                        dst = read_items(source, ef, e + start, count)
                    else:
                        pairs = read_items(source, ef, 2 * start, 2 * count).reshape(-1, 2)
                        src, dst = pairs[:, 0], pairs[:, 1]
                    if count and (src.min() < 0 or src.max() >= n or dst.min() < 0 or dst.max() >= n):
                        raise ValueError("edge ID outside [0,N)")
                    run = np.empty(count, dtype=RECORD)
                    run["src"], run["dst"] = src, dst
                    run["eid"] = np.arange(start, start + count, dtype=np.int64)
                    order = np.argsort(dst, kind="stable")
                    out.write(run[order].tobytes())
                paths.append(job.task("sort_00_{:08d}.bin".format(index), write_run))
        level = 1
        while len(paths) > 1:
            next_paths = []
            for index, start in enumerate(range(0, len(paths), fan_in)):
                group = paths[start:start + fan_in]
                next_paths.append(job.task("sort_{:02d}_{:08d}.bin".format(level, index),
                                           lambda out, group=group: merge(group, out)))
            paths, level = next_paths, level + 1
        ordered = paths[0]

        def indices(out):
            npy_header(out, (e,), "<i8")
            buffer = bytearray()
            for _, _, src in records(ordered):
                buffer.extend(struct.pack("<q", src))
                if len(buffer) >= 65536:
                    out.write(buffer)
                    buffer.clear()
            out.write(buffer)

        def indptr(out):
            npy_header(out, (n + 1,), "<i8")
            owner, consumed, buffer = 0, 0, bytearray()
            for dst, _, _ in records(ordered):
                while owner <= dst:
                    buffer.extend(struct.pack("<q", consumed))
                    owner += 1
                    if len(buffer) >= 65536:
                        out.write(buffer)
                        buffer.clear()
                consumed += 1
            while owner <= n:
                buffer.extend(struct.pack("<q", consumed))
                owner += 1
                if len(buffer) >= 65536:
                    out.write(buffer)
                    buffer.clear()
            out.write(buffer)

        job.task("original_indices.npy", indices)
        job.task("original_indptr.npy", indptr)
        return job.complete(["original_indptr.npy", "original_indices.npy"])


def reorder_features(features, mapping, workspace, batch_rows=16384,
                     memory_mib=256, stop_after=None):
    ff, mf = header(features), header(mapping)
    validate_features(ff)
    if len(mf["shape"]) != 1 or np.dtype(mf["dtype"]).kind != "i":
        raise ValueError("mapping must be signed integer [storage_rows]")
    row_bytes = ff["shape"][1] * np.dtype(ff["dtype"]).itemsize
    rows = mf["shape"][0]
    if batch_rows <= 0:
        raise ValueError("batch-rows must be positive")
    tasks = (rows + batch_rows - 1) // batch_rows
    heap = batch_rows * (row_bytes + 16) + row_bytes + 4 * MIB + 4096 * (tasks + 1)
    check_heap(heap, memory_mib)
    check_space(workspace, 2 * rows * row_bytes + 4096 * (tasks + 1))
    binding = dict(kind="feature_reorder", features=identity(features), mapping=identity(mapping),
                   batch_rows=batch_rows, memory_mib=memory_mib,
                   code_sha256=digest(__file__), numpy=np.__version__)
    if any(binding["features"][k] != v for k, v in ff.items()) or any(
            binding["mapping"][k] != v for k, v in mf.items()):
        raise ValueError("input header changed during preflight")
    with Job(workspace, binding, stop_after) as job:
        paths = []
        with Path(ff["path"]).open("rb") as source, Path(mf["path"]).open("rb") as ids:
            for index, start in enumerate(range(0, rows, batch_rows)):
                count = min(batch_rows, rows - start)
                def write_chunk(out, start=start, count=count):
                    nodes = read_items(ids, mf, start, count)
                    if nodes.size and (nodes.min() < -1 or nodes.max() >= ff["shape"][0]):
                        raise ValueError("mapping outside [-1,N)")
                    buffer = bytearray(count * row_bytes)
                    for i, node in enumerate(nodes):
                        if node >= 0:
                            source.seek(ff["offset"] + int(node) * row_bytes)
                            data = source.read(row_bytes)
                            if len(data) != row_bytes:
                                raise ValueError("short feature row")
                            buffer[i * row_bytes:(i + 1) * row_bytes] = data
                    out.write(buffer)
                paths.append(job.task("features_{:08d}.bin".format(index), write_chunk))

        def assemble(out):
            npy_header(out, (rows, ff["shape"][1]), ff["dtype"])
            for path in paths:
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(MIB), b""):
                        out.write(block)

        job.task("reordered_features.npy", assemble)
        return job.complete(["reordered_features.npy"])


def address_space_limit(mib):
    if mib < 128:
        raise ValueError("address-space-mib must be at least 128")
    requested = mib * MIB
    current_vm = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    if requested <= current_vm + 32 * MIB:
        raise ValueError("address-space limit leaves insufficient interpreter headroom")
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and requested > hard:
        raise ValueError("requested address-space limit exceeds inherited hard limit")
    resource.setrlimit(resource.RLIMIT_AS, (requested, hard))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "csc", "features", "status"])
    p.add_argument("--edges")
    p.add_argument("--features")
    p.add_argument("--mapping")
    p.add_argument("--workspace", required=True)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--slot-bytes", type=int, default=8192)
    p.add_argument("--replication-ratio", type=float, default=.2)
    p.add_argument("--chunk-edges", type=int, default=262144)
    p.add_argument("--fan-in", type=int, default=16)
    p.add_argument("--batch-rows", type=int, default=16384)
    p.add_argument("--memory-mib", type=int, default=256)
    p.add_argument("--address-space-mib", type=int, default=1024)
    p.add_argument("--stop-after-tasks", type=int)
    args = p.parse_args()
    if args.memory_mib <= 0 or (args.stop_after_tasks is not None and args.stop_after_tasks < 0):
        p.error("memory-mib must be positive and stop-after-tasks nonnegative")
    if args.command in ("plan", "csc") and (not args.edges or not args.features):
        p.error("plan/csc require --edges and --features")
    if args.command == "features" and (not args.mapping or not args.features):
        p.error("features requires --mapping and --features")
    def stop(signum, frame):
        global STOP
        STOP = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        if args.command == "status":
            result = _digit_paths.json_loads((Path(args.workspace) / "state.json").read_text())
            result = {k: v for k, v in result.items() if k not in ("binding", "files")}
            result["note"] = "Last durable state, not a process-liveness or integrity check"
        elif args.command == "plan":
            result = capacity_plan(args.edges, args.features, args.workspace, args.group_size,
                                   args.slot_bytes, args.replication_ratio, args.chunk_edges,
                                   args.fan_in, args.memory_mib)
        else:
            address_space_limit(args.address_space_mib)
            if args.command == "csc":
                result = build_csc(args.edges, args.features, args.workspace, args.chunk_edges,
                                   args.fan_in, args.memory_mib, args.stop_after_tasks)
            else:
                result = reorder_features(args.features, args.mapping, args.workspace,
                                          args.batch_rows, args.memory_mib, args.stop_after_tasks)
            result = {k: v for k, v in result.items() if k not in ("binding", "files")}
        print(json.dumps(result, indent=2, sort_keys=True))
    except Paused:
        print("Paused at a durable task boundary; repeat the SAME command to resume.")


if __name__ == "__main__":
    main()
