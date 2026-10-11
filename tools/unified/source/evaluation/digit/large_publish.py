"""Bounded complete-artifact validation and no-clobber atomic publication (Linux)."""

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
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sqlite3

import numpy as np

if __package__:
    from . import large_preprocess as lp, large_layout as ll
    from .io_geometry import IOGeometry
else:
    import large_preprocess as lp
    import large_layout as ll
    from io_geometry import IOGeometry

NAMES = ("group_members", "group_owner", "group_storage_base", "storage_to_node",
         "node_to_primary_row", "supernode_to_group", "reordered_indptr",
         "reordered_indices", "reordered_features")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_state(root, kind):
    state = _digit_paths.json_loads((root / "state.json").read_text())
    require(state["schema"] == lp.SCHEMA and state["phase"] == "complete"
            and state["binding"]["kind"] == kind, "incomplete/wrong stage: " + str(root))
    for name, item in state["files"].items():
        require(Path(name).name == name, "invalid checkpoint path")
        path = root / name
        require(not path.is_symlink() and path.is_file(), "invalid checkpoint file")
        require(path.stat().st_size == item["bytes"] and lp.digest(path) == item["sha256"],
                "checkpoint hash mismatch: " + name)
    return state


@contextlib.contextmanager
def accepted_stage(root, kind):
    root = Path(root).resolve(strict=True)
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        state = checked_state(root, kind)
        yield root, state
        current = _digit_paths.json_loads((root / "state.json").read_text())
        require(ll.contract_hash(current) == ll.contract_hash(state), "stage changed during publication")


def rename_noreplace(source, target):
    # Atomic directory rename must never replace even an empty competing target.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))
    lp.fsync_dir(target.parent)


class Arrays:
    def __init__(self, root, stack):
        self.info, self.files = {}, {}
        for name in NAMES:
            path = root / (name + ".npy")
            require(not path.is_symlink(), "array symlink forbidden")
            self.info[name] = lp.header(path)
            self.files[name] = stack.enter_context(path.open("rb"))

    def get(self, name, start, count=1):
        info = self.info[name]
        length = math.prod(info["shape"])
        require(0 <= start <= length and 0 <= count <= length - start, "array read outside bounds: " + name)
        return lp.read_items(self.files[name], info, start, count)

    def one(self, name, start):
        return int(self.get(name, start)[0])


@contextlib.contextmanager
def audit_file(root):
    # One explicitly owned, disposable file; SIGKILL leaves it for this same
    # workspace's next validation attempt, never an accumulating temp tree.
    state = _digit_paths.json_loads((root / "state.json").read_text())
    require(state["binding"]["kind"] == "digit-stream-publication-v1", "audit requires owned workspace")
    path = root / ".validation-audit.sqlite"
    require(not path.is_symlink() and (not path.exists() or path.is_file()), "invalid audit scratch")
    if path.exists():
        path.unlink()
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


def pause_requested():
    if lp.STOP:
        raise lp.Paused("validation interrupted; restart its bounded scan")


def validate(root, manifest, sources, chunk_rows=4096, memory_mib=64):
    """Full exact scan, including feature bytes and expanded adjacency multiplicity.

    SQLite state lives in a bounded-cache, disk-backed private scratch directory.
    No graph-sized NumPy array/mmap or degree-sized expansion is allocated.
    """
    root = Path(root)
    n, e = manifest["dataset"]["num_nodes"], manifest["dataset"]["num_edges"]
    grouping = manifest["grouping"]
    g, size, primary = (grouping[k] for k in ("num_groups", "group_size", "num_primary_groups"))
    rows, dim = (manifest["feature"][k] for k in ("num_storage_rows", "dim"))
    geo = IOGeometry.from_dict(manifest["io_geometry"])
    align = geo.rows_per_slot
    require(chunk_rows > 0 and n > 0 and e >= 0 and rows > 0, "invalid validation sizes")
    require(0 <= primary <= g and grouping["num_replica_groups"] == g - primary,
            "invalid primary/replica counts")
    ratio = grouping["replication_ratio"]
    require(math.isfinite(ratio) and ratio >= 0 and (g - primary) * size <= math.floor(ratio * n),
            "replication budget exceeded")
    require(manifest["schema_name"] == "digit-artifact-bundle" and manifest["schema_version"] == 2,
            "expected schema-v2")
    require(geo.semantic_sha256() == manifest["io_geometry_sha256"] and geo.group_size == size,
            "geometry hash/group mismatch")
    require(manifest["io"] == dict(page_size=geo.cache_slot_bytes, alignment_rows=align)
            and grouping["supernode_id_start"] == n, "geometry/ID metadata mismatch")
    require(manifest["graph"] == dict(format="csc", num_logical_nodes=n, num_supernodes=g,
                                     num_graph_nodes=n + g), "graph metadata mismatch")
    lp.check_heap(8 * lp.MIB + chunk_rows * 64 + (dim * 4) * 3 + align * 16, memory_mib)
    # Hard page ceiling below this reserved upper envelope; SQLITE_FULL fails closed.
    audit_bytes = max(4 * lp.MIB, 256 * (e + n + g * size + 1))
    lp.check_space(root, audit_bytes)
    for identity in sources.values():
        require(lp.identity(identity["path"]) == identity, "validation source identity changed")
    with contextlib.ExitStack() as stack:
        arrays = Arrays(root, stack)
        shapes = dict(group_members=[g, size], group_owner=[g], group_storage_base=[g],
                      storage_to_node=[rows], node_to_primary_row=[n], supernode_to_group=[g],
                      reordered_indptr=[n + g + 1], reordered_indices=[e - g * (size - 1)],
                      reordered_features=[rows, dim])
        for name in NAMES:
            info = arrays.info[name]
            dtype = "<f4" if name == "reordered_features" else "<i8"
            require(info["shape"] == shapes[name] and info["dtype"] == dtype
                    and not info["fortran_order"], "array shape/dtype mismatch: " + name)
            expected = dict(path=name + ".npy", dtype=dtype, shape=shapes[name],
                            sha256=lp.digest(info["path"]))
            require(manifest["files"][name] == expected, "manifest array/hash mismatch: " + name)
        require(manifest["feature"]["dtype"] == "<f4" and manifest["feature"]["row_bytes"] == dim * 4
                and geo.feature_row_bytes == dim * 4 and rows % align == 0
                and manifest["feature"]["padding_node_id"] == -1, "feature geometry mismatch")
        ptr, idx, features = (sources[k] for k in ("indptr", "indices", "features"))
        require(ptr["shape"] == [n + 1] and idx["shape"] == [e]
                and np.dtype(ptr["dtype"]).kind in "iu" and np.dtype(idx["dtype"]).kind in "iu"
                and features["shape"] == [n, dim] and features["dtype"] == "<f4"
                and not features["fortran_order"], "source shape mismatch")
        pf = stack.enter_context(Path(ptr["path"]).open("rb"))
        inf = stack.enter_context(Path(idx["path"]).open("rb"))
        ff = stack.enter_context(Path(features["path"]).open("rb"))
        scratch = stack.enter_context(audit_file(root))
        db = stack.enter_context(contextlib.closing(sqlite3.connect(str(scratch))))
        db.execute("PRAGMA page_size=4096")
        db.execute("PRAGMA max_page_count={}".format(audit_bytes // 4096))
        db.execute("PRAGMA journal_mode=OFF")  # Disposable scratch, never a checkpoint.
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA mmap_size=0")
        db.execute("PRAGMA cache_size=-2048")
        db.execute("PRAGMA temp_store=FILE")
        db.executescript("CREATE TABLE hot(node INTEGER PRIMARY KEY);"
                         "CREATE TABLE primary_node(node INTEGER PRIMARY KEY,row INTEGER);"
                         "CREATE TABLE seen(gid INTEGER PRIMARY KEY);"
                         "CREATE TABLE edges(owner INTEGER,node INTEGER,balance INTEGER,"
                         "PRIMARY KEY(owner,node)) WITHOUT ROWID;")
        if "hot_nodes" in sources:
            hot = sources["hot_nodes"]
            kind = np.dtype(hot["dtype"]).kind
            require(len(hot["shape"]) == 1 and kind in "iub", "invalid hot source")
            require(kind != "b" or hot["shape"] == [n], "invalid hot mask length")
            hf = stack.enter_context(Path(hot["path"]).open("rb"))
            for start in range(0, hot["shape"][0], chunk_rows):
                values = lp.read_items(hf, hot, start, min(chunk_rows, hot["shape"][0] - start))
                nodes = np.flatnonzero(values) + start if kind == "b" else values
                for node in nodes:
                    require(0 <= int(node) < n, "invalid hot node")
                    db.execute("INSERT OR IGNORE INTO hot VALUES(?)", (int(node),))
        for gid in range(g):
            pause_requested()
            require(arrays.one("supernode_to_group", gid) == gid, "supernode group identity mismatch")
            require(arrays.one("group_storage_base", gid) == gid * align, "noncanonical group base")
            require(0 <= arrays.one("group_owner", gid) < n, "invalid group owner")
            members = arrays.get("group_members", gid * size, size)
            require(len(set(int(v) for v in members)) == size, "duplicate group member")
            for local, member in enumerate(members):
                node, row = int(member), gid * align + local
                require(0 <= node < n and arrays.one("storage_to_node", row) == node,
                        "group/storage mismatch")
                require(db.execute("SELECT 1 FROM hot WHERE node=?", (node,)).fetchone() is None,
                        "hot node in group")
                if gid < primary:
                    db.execute("INSERT INTO primary_node VALUES(?,?)", (node, row))
                else:
                    require(db.execute("SELECT 1 FROM primary_node WHERE node=?", (node,)).fetchone(),
                            "replica of non-primary node")
            for local in range(size, align):
                require(arrays.one("storage_to_node", gid * align + local) == -1, "group padding used")
        hot_count = db.execute("SELECT count(*) FROM hot").fetchone()[0]
        hot_start = g * align
        cold_start = ((hot_start + hot_count + align - 1) // align) * align
        cold_count = n - primary * size - hot_count
        require(cold_count >= 0 and rows == ((cold_start + cold_count + align - 1) // align) * align,
                "storage row count inconsistent")
        for hot_partition, start, count in ((True, hot_start, hot_count), (False, cold_start, cold_count)):
            previous = -1
            for row in range(start, start + count):
                node = arrays.one("storage_to_node", row)
                require(previous < node < n, "raw partition not ascending/unique")
                previous = node
                require(db.execute("SELECT 1 FROM primary_node WHERE node=?", (node,)).fetchone() is None,
                        "primary member repeated in raw partition")
                is_hot = db.execute("SELECT 1 FROM hot WHERE node=?", (node,)).fetchone() is not None
                require(is_hot == hot_partition, "raw hot/cold partition mismatch")
            end = cold_start if hot_partition else rows
            for row in range(start + count, end):
                require(arrays.one("storage_to_node", row) == -1, "raw padding used")
        for node in range(n):
            row = arrays.one("node_to_primary_row", node)
            require(0 <= row < rows and arrays.one("storage_to_node", row) == node, "inverse mismatch")
            primary_row = db.execute("SELECT row FROM primary_node WHERE node=?", (node,)).fetchone()
            require(row == primary_row[0] if primary_row else row >= hot_start,
                    "inverse points at a replica")
        # Exact feature bytes, including NaN payloads and zero padding.
        feature_file = arrays.files["reordered_features"]
        feature_file.seek(arrays.info["reordered_features"]["offset"])
        zero = bytes(dim * 4)
        for start in range(0, rows, chunk_rows):
            pause_requested()
            for node in arrays.get("storage_to_node", start, min(chunk_rows, rows - start)):
                require(-1 <= int(node) < n, "storage ID out of bounds")
                expected = zero
                if node >= 0:
                    ff.seek(features["offset"] + int(node) * dim * 4)
                    expected = ff.read(dim * 4)
                require(len(expected) == dim * 4 and feature_file.read(dim * 4) == expected,
                        "feature payload bit mismatch")
        # Count original edge multiplicities on disk, then consume expanded edges.
        previous = 0
        for owner in range(n):
            pause_requested()
            begin, end = (int(v) for v in lp.read_items(pf, ptr, owner, 2))
            require(begin == previous and begin <= end <= e, "invalid original CSC pointer")
            previous = end
            for start in range(begin, end, chunk_rows):
                pause_requested()
                for node in lp.read_items(inf, idx, start, min(chunk_rows, end - start)):
                    require(0 <= int(node) < n, "invalid original edge")
                    db.execute("INSERT INTO edges VALUES(?,?,1) ON CONFLICT(owner,node) "
                               "DO UPDATE SET balance=balance+1", (owner, int(node)))
        require(previous == e, "original CSC terminal mismatch")
        previous, expanded = 0, 0
        rewritten_e = shapes["reordered_indices"][0]
        for owner in range(n + g):
            pause_requested()
            begin, end = (int(v) for v in arrays.get("reordered_indptr", owner, 2))
            require(begin == previous and begin <= end <= rewritten_e, "invalid rewritten pointer")
            require(owner < n or begin == end, "supernode column is not empty")
            previous = end
            last_group = -1
            for start in range(begin, end, chunk_rows):
                pause_requested()
                for value in arrays.get("reordered_indices", start, min(chunk_rows, end - start)):
                    node = int(value)
                    require(0 <= node < n + g, "invalid rewritten edge")
                    if node < n:
                        require(last_group == -1, "raw edge follows group edge")
                        members = (node,)
                    else:
                        gid = node - n
                        require(gid > last_group and arrays.one("group_owner", gid) == owner,
                                "group order/owner mismatch")
                        last_group = gid
                        db.execute("INSERT INTO seen VALUES(?)", (gid,))
                        members = arrays.get("group_members", gid * size, size)
                    for member in members:
                        cursor = db.execute("UPDATE edges SET balance=balance-1 WHERE owner=? AND node=? "
                                            "AND balance>0", (owner, int(member)))
                        require(cursor.rowcount == 1, "expanded adjacency mismatch")
                        expanded += 1
        require(previous == rewritten_e and expanded == e, "expanded edge count mismatch")
        require(db.execute("SELECT 1 FROM edges WHERE balance!=0 LIMIT 1").fetchone() is None,
                "original edges missing from expansion")
        require(db.execute("SELECT count(*) FROM seen").fetchone()[0] == g, "unreferenced group")
        used_bytes = db.execute("PRAGMA page_count").fetchone()[0] * 4096
    lp.inputs_unchanged(sources)
    return dict(passed=True, schema="digit-stream-bundle-validation-v1", feature_bit_exact=True,
                expanded_adjacency_exact=True, edges=e, groups=g, storage_rows=rows,
                audit_database_bytes=used_bytes, audit_database_limit_bytes=audit_bytes,
                gpu_accessed=False, raw_ssd_accessed=False)


def publish(group_workspace, layout_workspace, rewrite_workspace, feature_workspace, output,
            dataset_name="IGB", dataset_size="small", slot_bytes=8192, minimum_transfer_bytes=4096,
            chunk_rows=4096, memory_mib=64, stop_after_tasks=None, stop_before_publish=False):
    # Do not resolve the final component: a pre-existing symlink is not our target.
    requested = Path(output).absolute()
    require(requested.name not in ("", ".", ".."), "invalid output name")
    requested.parent.mkdir(parents=True, exist_ok=True)
    target = requested.parent.resolve() / requested.name
    staging = target.with_name(target.name + ".building")
    require(not target.is_symlink() and not staging.is_symlink(), "publication symlink forbidden")
    metadata_paths = [Path(group_workspace) / "binding.json", Path(group_workspace) / "exports/state.json"]
    metadata_paths += [Path(path) / "state.json" for path in (layout_workspace, rewrite_workspace, feature_workspace)]
    metadata_bytes = sum(path.stat().st_size for path in metadata_paths)
    # Bound decoding of upstream checkpoint dictionaries before loading them.
    lp.check_heap(8 * lp.MIB + metadata_bytes * 16 + max(0, chunk_rows) * 64, memory_mib)
    with contextlib.ExitStack() as stack:
        lock = stack.enter_context(target.with_name(target.name + ".publish.lock").open("a+b"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, source, progress, group_contract, db_hash = stack.enter_context(ll.accepted_groups(group_workspace))
        lr, ls = stack.enter_context(accepted_stage(layout_workspace, "digit-stream-layout-v1"))
        rr, rs = stack.enter_context(accepted_stage(rewrite_workspace, "digit-stream-rewrite-v1"))
        fr, fs = stack.enter_context(accepted_stage(feature_workspace, "feature_reorder"))
        for stage in (ls, rs):
            require(stage["binding"]["source_export_contract_sha256"] == group_contract
                    and stage["binding"]["database_sha256"] == db_hash,
                    "mixed grouping contracts")
        sources = {key: value for key, value in source["sources"].items() if value is not None}
        require(rs["binding"]["indptr"] == sources["indptr"] and rs["binding"]["indices"] == sources["indices"],
                "rewrite CSC source mismatch")
        sources["features"] = fs["binding"]["features"]
        for identity in sources.values():
            require(lp.identity(identity["path"]) == identity, "publication source identity changed")
        require(fs["binding"]["mapping"] == lp.identity(lr / "storage_to_node.npy"),
                "feature/layout mapping mismatch")
        n, e = sources["indptr"]["shape"][0] - 1, sources["indices"]["shape"][0]
        feature = sources["features"]
        lp.validate_features(feature)
        require(feature["shape"][0] == n and feature["dtype"] == "<f4", "requires float32 [N,D] features")
        size = source["config"]["group_size"]
        g = progress["primary_groups"] + progress["replica_groups"]
        geo = IOGeometry.create(feature_row_bytes=feature["shape"][1] * 4, group_size=size,
                               minimum_transfer_bytes=minimum_transfer_bytes, target_request_bytes=slot_bytes)
        require(geo.cache_slot_bytes == slot_bytes and ls["binding"]["alignment_rows"] == geo.rows_per_slot,
                "layout/geometry mismatch")
        binding = dict(kind="digit-stream-publication-v1", output=str(target),
                       groups=group_contract, layout=ll.contract_hash(ls), rewrite=ll.contract_hash(rs),
                       features=ll.contract_hash(fs), source=sources, geometry=geo.to_dict(),
                       dataset_name=dataset_name, dataset_size=dataset_size,
                       chunk_rows=chunk_rows, memory_mib=memory_mib, source_sha256=lp.digest(__file__),
                       foundation_sha256=lp.digest(lp.__file__), layout_sha256=lp.digest(ll.__file__),
                       geometry_sha256=lp.digest(Path(__file__).with_name("io_geometry.py")), numpy=np.__version__)
        if target.exists():
            done = checked_state(target, binding["kind"])
            require(done["binding"] == binding and done.get("complete_artifact") is True,
                    "published target belongs to different inputs")
            return done
        inputs = {name: (Path(group_workspace).resolve() / "exports" / (name + ".npy"))
                  for name in ("group_members", "group_owner")}
        inputs.update({name: lr / (name + ".npy") for name in
                       ("group_storage_base", "storage_to_node", "node_to_primary_row")})
        inputs.update({name: rr / (name + ".npy") for name in ("reordered_indptr", "reordered_indices")})
        inputs["reordered_features"] = fr / "reordered_features.npy"
        require(chunk_rows > 0, "chunk_rows must be positive")
        lp.check_heap(8 * lp.MIB + metadata_bytes * 16 + chunk_rows * 64
                      + feature["shape"][1] * 12 + geo.rows_per_slot * 16,
                      memory_mib)
        lp.check_space(staging, sum(p.stat().st_size for p in inputs.values()) + 8 * g + lp.MIB
                       + max(4 * lp.MIB, 256 * (e + n + g * size + 1)))
        with lp.Job(staging, binding, stop_after_tasks) as job:
            for name, path in inputs.items():
                expected_hash = lp.digest(path)
                def copy(out, path=path, expected_hash=expected_hash):
                    hasher = lp.hashlib.sha256()
                    with path.open("rb") as src:
                        for data in iter(lambda: src.read(lp.MIB), b""):
                            hasher.update(data)
                            out.write(data)
                    require(hasher.hexdigest() == expected_hash, "source changed during copy")
                job.task(name + ".npy", copy)
                require(job.state["files"][name + ".npy"]["sha256"] == expected_hash, "copy hash mismatch")
            def supernodes(out):
                lp.npy_header(out, (g,), "<i8")
                for start in range(0, g, chunk_rows):
                    out.write(np.arange(start, min(start + chunk_rows, g), dtype="<i8").tobytes())
            job.task("supernode_to_group.npy", supernodes)
            headers = {name: lp.header(staging / (name + ".npy")) for name in NAMES}
            manifest = dict(schema_name="digit-artifact-bundle", schema_version=2,
                dataset=dict(name=dataset_name, size=dataset_size, num_nodes=n, num_edges=e),
                feature=dict(dim=feature["shape"][1], dtype="<f4", row_bytes=feature["shape"][1] * 4,
                             num_storage_rows=headers["storage_to_node"]["shape"][0], padding_node_id=-1),
                grouping=dict(group_size=size, replication_ratio=source["config"]["ratio"], num_groups=g,
                              num_primary_groups=progress["primary_groups"], num_replica_groups=progress["replica_groups"],
                              supernode_id_start=n),
                io=dict(page_size=slot_bytes, alignment_rows=geo.rows_per_slot),
                io_geometry=geo.to_dict(), io_geometry_sha256=geo.semantic_sha256(),
                graph=dict(format="csc", num_logical_nodes=n, num_supernodes=g, num_graph_nodes=n + g),
                files={name: dict(path=name + ".npy", dtype=headers[name]["dtype"], shape=headers[name]["shape"],
                                 sha256=job.state["files"][name + ".npy"]["sha256"]) for name in NAMES},
                source=sources, metadata=dict(preprocessing="streamed-v1", publication_binding=binding))
            def validation(out):
                report = validate(staging, manifest, sources, chunk_rows, memory_mib)
                out.write((json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
            job.task("validation.json", validation)
            job.task("manifest.json", lambda out: out.write((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()))
            lp.inputs_unchanged(sources)
            # Recheck durable contracts before crossing the publication boundary.
            for root, state in ((lr, ls), (rr, rs), (fr, fs)):
                require(ll.contract_hash(_digit_paths.json_loads((root / "state.json").read_text())) == ll.contract_hash(state),
                        "source stage changed")
            require(lp.digest(Path(group_workspace) / "groups.sqlite") == db_hash
                    and ll.contract_hash(_digit_paths.json_loads((Path(group_workspace) / "exports/state.json").read_text())) == group_contract,
                    "group source changed")
            job.complete([name + ".npy" for name in NAMES] + ["validation.json", "manifest.json"])
            job.state["complete_artifact"] = True
            job.save()
            result = job.state
        if stop_before_publish or lp.STOP:
            raise lp.Paused("validated staging retained; publication not performed")
        rename_noreplace(staging, target)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("group-workspace", "layout-workspace", "rewrite-workspace", "feature-workspace", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--dataset-name", default="IGB")
    p.add_argument("--dataset-size", default="small")
    p.add_argument("--slot-bytes", type=int, default=8192)
    p.add_argument("--minimum-transfer-bytes", type=int, default=4096)
    p.add_argument("--chunk-rows", type=int, default=4096)
    p.add_argument("--memory-mib", type=int, default=64)
    p.add_argument("--address-space-mib", type=int, default=1024)
    p.add_argument("--stop-after-tasks", type=int)
    p.add_argument("--stop-before-publish", action="store_true")
    args = vars(p.parse_args())
    lp.address_space_limit(args.pop("address_space_mib"))
    def stop(signum, frame):
        lp.STOP = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        publish(**args)
        print("Validated schema-v2 artifact atomically published: " + args["output"])
    except lp.Paused:
        print("Paused; repeat the same settings to resume (no unverified artifact published)")


if __name__ == "__main__":
    main()
