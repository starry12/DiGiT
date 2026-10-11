"""Stream layout maps from an accepted external-group workspace. No SSD I/O."""

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
import contextlib
import fcntl
import json
from pathlib import Path
import signal
import sqlite3

import numpy as np

if __package__:
    from . import large_preprocess as lp
else:
    import large_preprocess as lp


def raw_query(hot):
    return ("SELECT owner FROM owners WHERE NOT EXISTS(SELECT 1 FROM used WHERE node=owner) AND "
            + ("EXISTS" if hot else "NOT EXISTS")
            + "(SELECT 1 FROM hot WHERE node=owner) ORDER BY owner")


def contract_hash(state):
    # An idempotent upstream recheck may refresh PID/time/RSS, not its outputs.
    # Bind the durable contract, not those volatile observations.
    contract = {k: state[k] for k in ("schema", "phase", "binding", "files", "outputs")}
    contract["outputs"] = sorted(contract["outputs"])
    return lp.hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextlib.contextmanager
def accepted_groups(root):
    root = Path(root).resolve(strict=True)
    # Same lock as grouping writer; never modify the source database or exports.
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = root / "exports/state.json"
        state = _digit_paths.json_loads(manifest.read_text())
        binding = _digit_paths.json_loads((root / "binding.json").read_text())
        required = {"group_members.npy", "group_owner.npy", "owner_order.npy", "eligible_degree.npy",
                    "primary_grouped_nodes.npy", "covered_edge_mask.npy", "statistics.json"}
        if binding.get("schema") != "digit-external-grouping-v1" or not required.issubset(state["files"]):
            raise ValueError("unsupported/incomplete group export contract")
        if state["phase"] != "complete" or state["binding"]["grouping_binding"] != binding:
            raise ValueError("group exports incomplete or binding mismatch")
        database = root / "groups.sqlite"
        if lp.digest(database) != state["binding"]["database_sha256"]:
            raise ValueError("group database hash mismatch")
        for name, desc in state["files"].items():
            if Path(name).name != name:
                raise ValueError("invalid export path")
            path = root / "exports" / name
            if path.stat().st_size != desc["bytes"] or lp.digest(path) != desc["sha256"]:
                raise ValueError("group export hash mismatch: " + name)
        with contextlib.closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("PRAGMA mmap_size=0")
            db.execute("PRAGMA cache_size=-2048")
            db.execute("PRAGMA query_only=ON")
            saved = _digit_paths.json_loads(db.execute("SELECT value FROM meta WHERE key='binding'").fetchone()[0])
            progress = _digit_paths.json_loads(db.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0])
            if saved != binding or progress["phase"] != "grouping_complete":
                raise ValueError("group database incomplete or binding mismatch")
            yield db, binding, progress, contract_hash(state), state["binding"]["database_sha256"]


def build(group_workspace, workspace, alignment_rows, chunk_rows=65536, fan_in=16,
          memory_mib=64, stop_after_tasks=None):
    if alignment_rows <= 0 or chunk_rows <= 0 or fan_in < 2:
        raise ValueError("invalid alignment/chunk/fan-in")
    with accepted_groups(group_workspace) as (db, source, progress, manifest_sha, database_sha):
        n = source["sources"]["indptr"]["shape"][0] - 1
        size = source["config"]["group_size"]
        primary = progress["primary_groups"]
        total = primary + progress["replica_groups"]
        if size > alignment_rows or size > chunk_rows:
            raise ValueError("group must fit alignment and chunk buffer")
        if db.execute("SELECT count(*) FROM used JOIN hot USING(node)").fetchone()[0]:
            raise ValueError("hot node is primary grouped")
        hot = db.execute("SELECT count(*) FROM hot").fetchone()[0]
        cold = n - primary * size - hot
        if cold < 0:
            raise ValueError("invalid node accounting")
        hot_start = total * alignment_rows
        cold_start = ((hot_start + hot + alignment_rows - 1) // alignment_rows) * alignment_rows
        rows = ((cold_start + cold + alignment_rows - 1) // alignment_rows) * alignment_rows
        if max(rows, total, n) >= 2 ** 63:
            raise ValueError("layout exceeds int64 range")
        groups_per_chunk = max(1, chunk_rows // size)
        run_count = (primary + groups_per_chunk - 1) // groups_per_chunk + 2
        files, levels, count = run_count, 1, run_count
        while count > 1:
            count = (count + fan_in - 1) // fan_in
            files += count
            levels += 1
        lp.check_heap(lp.sort_heap_bound(chunk_rows, fan_in, files + 4) + 2 * lp.MIB, memory_mib)
        # Retain every inverse-sort generation, plus the three final maps.
        disk = 24 * n * levels + 8 * (total + rows + n) + 4096 * (files + 4)
        lp.check_space(workspace, disk)
        binding = dict(kind="digit-stream-layout-v1", source_workspace=str(Path(group_workspace).resolve()),
                       source_export_contract_sha256=manifest_sha, database_sha256=database_sha,
                       alignment_rows=alignment_rows, chunk_rows=chunk_rows, fan_in=fan_in,
                       memory_mib=memory_mib, source_sha256=lp.digest(__file__),
                       foundation_sha256=lp.digest(lp.__file__), numpy=np.__version__)
        with lp.Job(workspace, binding, stop_after_tasks) as job:
            def bases(out):
                lp.npy_header(out, (total,), "<i8")
                for start in range(0, total, chunk_rows):
                    out.write((np.arange(start, min(start + chunk_rows, total), dtype="<i8")
                               * alignment_rows).tobytes())

            def storage(out):
                lp.npy_header(out, (rows,), "<i8")
                buffer = bytearray()
                def emit(value):
                    buffer.extend(lp.struct.pack("<q", value))
                    if len(buffer) >= 65536:
                        out.write(buffer)
                        buffer.clear()
                seen = 0
                for gid, raw in db.execute("SELECT gid,members FROM groups ORDER BY gid"):
                    values = np.frombuffer(raw, dtype="<i8")
                    if gid != seen or len(values) != size:
                        raise ValueError("invalid group shape/order")
                    for v in values:
                        emit(int(v))
                    for _ in range(alignment_rows - size):
                        emit(-1)
                    seen += 1
                if seen != total:
                    raise ValueError("group count mismatch")
                for node, in db.execute(raw_query(True)):
                    emit(node)
                for _ in range(cold_start - hot_start - hot):
                    emit(-1)
                for node, in db.execute(raw_query(False)):
                    emit(node)
                for _ in range(rows - cold_start - cold):
                    emit(-1)
                out.write(buffer)

            job.task("group_storage_base.npy", bases)
            job.task("storage_to_node.npy", storage)
            runs = []
            for index, start in enumerate(range(0, primary, groups_per_chunk)):
                end = min(start + groups_per_chunk, primary)
                def primary_run(out, start=start, end=end):
                    records = np.empty((end - start) * size, dtype=lp.RECORD)
                    cursor = 0
                    for gid, raw in db.execute("SELECT gid,members FROM groups WHERE gid>=? AND gid<? ORDER BY gid",
                                               (start, end)):
                        values = np.frombuffer(raw, dtype="<i8")
                        if len(values) != size:
                            raise ValueError("invalid member shape")
                        records["dst"][cursor:cursor + size] = values
                        mapped = gid * alignment_rows + np.arange(size, dtype=np.int64)
                        records["eid"][cursor:cursor + size] = mapped
                        records["src"][cursor:cursor + size] = mapped
                        cursor += size
                    if cursor != len(records):
                        raise ValueError("missing primary group")
                    out.write(records[np.argsort(records["dst"], kind="stable")].tobytes())
                runs.append(job.task("primary_{:08d}.bin".format(index), primary_run))
            for is_hot, first_row, expected in ((True, hot_start, hot), (False, cold_start, cold)):
                def raw_run(out, is_hot=is_hot, first_row=first_row, expected=expected):
                    count, buffer = 0, bytearray()
                    for node, in db.execute(raw_query(is_hot)):
                        row = first_row + count
                        buffer.extend(lp.struct.pack("<qqq", node, row, row))
                        count += 1
                        if len(buffer) >= 65536:
                            out.write(buffer)
                            buffer.clear()
                    out.write(buffer)
                    if count != expected:
                        raise ValueError("raw partition count mismatch")
                runs.append(job.task("raw_hot.bin" if is_hot else "raw_cold.bin", raw_run))
            level = 0
            while len(runs) > 1:
                following = []
                for index, start in enumerate(range(0, len(runs), fan_in)):
                    parts = runs[start:start + fan_in]
                    following.append(job.task("inverse_{:03d}_{:08d}.bin".format(level, index),
                                              lambda out, parts=parts: lp.merge(parts, out)))
                runs, level = following, level + 1

            def inverse(out):
                lp.npy_header(out, (n,), "<i8")
                count, buffer = 0, bytearray()
                mapping = lp.header(job.root / "storage_to_node.npy")
                with (job.root / "storage_to_node.npy").open("rb") as source_file:
                    for node, row, duplicate in lp.records(runs[0]):
                        if node != count or row != duplicate or not 0 <= row < rows:
                            raise ValueError("missing/duplicate logical node or invalid primary row")
                        if int(lp.read_items(source_file, mapping, row, 1)[0]) != node:
                            raise ValueError("primary inverse disagrees with storage map")
                        buffer.extend(lp.struct.pack("<q", row))
                        count += 1
                        if len(buffer) >= 65536:
                            out.write(buffer)
                            buffer.clear()
                out.write(buffer)
                if count != n:
                    raise ValueError("incomplete primary inverse")
            job.task("node_to_primary_row.npy", inverse)
            summary = dict(num_nodes=n, group_size=size, num_groups=total, num_primary_groups=primary,
                           alignment_rows=alignment_rows, num_storage_rows=rows,
                           hot_start=hot_start, hot_count=hot, cold_start=cold_start, cold_count=cold,
                           retained_scratch_reservation_bytes=disk, full_artifact=False)
            job.task("layout.json", lambda out: out.write((json.dumps(summary, sort_keys=True, indent=2) + "\n").encode()))
            if lp.digest(Path(group_workspace) / "groups.sqlite") != database_sha or contract_hash(
                    _digit_paths.json_loads((Path(group_workspace) / "exports/state.json").read_text())) != manifest_sha:
                raise ValueError("group source changed while building layout")
            return job.complete(["group_storage_base.npy", "storage_to_node.npy", "node_to_primary_row.npy", "layout.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-workspace", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--alignment-rows", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--fan-in", type=int, default=16)
    parser.add_argument("--memory-mib", type=int, default=64)
    parser.add_argument("--address-space-mib", type=int, default=1024)
    parser.add_argument("--stop-after-tasks", type=int)
    args = parser.parse_args()
    def stop(signum, frame):
        lp.STOP = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    lp.address_space_limit(args.address_space_mib)
    try:
        build(args.group_workspace, args.workspace, args.alignment_rows, args.chunk_rows,
              args.fan_in, args.memory_mib, args.stop_after_tasks)
        print("Streaming storage layout completed (not a complete artifact)")
    except lp.Paused:
        print("Paused; repeat the same configuration to resume")


if __name__ == "__main__":
    main()
