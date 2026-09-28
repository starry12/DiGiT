"""Exact degree-bounded adjacency audit with paged bits; no per-edge database."""

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
import contextlib
from collections import OrderedDict
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from . import large_publish as old, large_preprocess as lp
else:
    import large_publish as old
    import large_preprocess as lp


class Bits:
    """Disposable file-backed bitmap with a fixed 512-KiB page cache."""
    def __init__(self, path, count):
        self.path, self.count = Path(path), count
        if self.path.is_symlink():
            raise ValueError("bitmap symlink")
        self.file = self.path.open("w+b")
        self.file.truncate((count + 7) // 8)
        self.cache = OrderedDict()

    def page(self, index):
        if not 0 <= index < self.count:
            raise ValueError("bitmap ID outside range")
        key = index // 32768
        if key not in self.cache:
            if len(self.cache) >= 128:
                removed, data = self.cache.popitem(last=False)
                self.file.seek(removed * 4096)
                self.file.write(data)
            self.file.seek(key * 4096)
            self.cache[key] = bytearray(self.file.read(min(4096, (self.count + 7) // 8 - key * 4096)))
        self.cache.move_to_end(key)
        return self.cache[key]

    def get(self, index):
        return bool(self.page(index)[(index // 8) % 4096] & (1 << (index % 8)))

    def set(self, index):
        value = self.get(index)
        self.page(index)[(index // 8) % 4096] |= 1 << (index % 8)
        return value

    def close(self):
        self.file.close()
        self.path.unlink()  # Only this validator's owned disposable scratch.


def validate(root, manifest, sources, chunk_rows=4096, memory_mib=64):
    require = old.require
    root = Path(root)
    n, e = manifest["dataset"]["num_nodes"], manifest["dataset"]["num_edges"]
    group = manifest["grouping"]
    g, size, primary = (group[k] for k in ("num_groups", "group_size", "num_primary_groups"))
    rows, dim = (manifest["feature"][k] for k in ("num_storage_rows", "dim"))
    cap = manifest["metadata"]["validation_degree_cap"]
    require(cap > 0 and chunk_rows > 0 and n > 0, "invalid cap/size")
    geo = old.IOGeometry.from_dict(manifest["io_geometry"])
    align = geo.rows_per_slot
    lp.check_heap(8 * lp.MIB + 64 * cap + chunk_rows * 64 + dim * 12 + align * 16, memory_mib)
    scratch = 2 * ((n + 7) // 8) + (g + 7) // 8
    lp.check_space(root, scratch + lp.MIB)
    require(manifest["schema_name"] == "digit-artifact-bundle" and manifest["schema_version"] == 2,
            "schema mismatch")
    require(geo.semantic_sha256() == manifest["io_geometry_sha256"] and geo.group_size == size,
            "geometry mismatch")
    require(manifest["io"] == dict(page_size=geo.cache_slot_bytes, alignment_rows=align)
            and group["supernode_id_start"] == n, "I/O or IDs mismatch")
    require(manifest["graph"] == dict(format="csc", num_logical_nodes=n, num_supernodes=g, num_graph_nodes=n + g),
            "graph metadata mismatch")
    ratio = group["replication_ratio"]
    require(math.isfinite(ratio) and ratio >= 0 and 0 <= primary <= g and group["num_replica_groups"] == g - primary
            and (g - primary) * size <= math.floor(ratio * n), "replica budget mismatch")
    for info in sources.values():
        require(lp.identity(info["path"]) == info, "source identity changed")
    state = _digit_paths.json_loads((root / "state.json").read_text())
    require(state["binding"]["kind"] == "digit-stream-publication-v1", "scratch requires owned publication stage")
    with contextlib.ExitStack() as stack:
        arrays = old.Arrays(root, stack)
        shapes = dict(group_members=[g, size], group_owner=[g], group_storage_base=[g], storage_to_node=[rows],
                      node_to_primary_row=[n], supernode_to_group=[g], reordered_indptr=[n + g + 1],
                      reordered_indices=[e - g * (size - 1)], reordered_features=[rows, dim])
        for name in old.NAMES:
            h = arrays.info[name]
            dtype = "<f4" if name == "reordered_features" else "<i8"
            require(h["shape"] == shapes[name] and h["dtype"] == dtype and not h["fortran_order"], "array metadata mismatch")
            require(manifest["files"][name] == dict(path=name + ".npy", shape=shapes[name], dtype=dtype,
                                                  sha256=lp.digest(h["path"])), "array hash mismatch")
        require(rows > 0 and rows % align == 0 and manifest["feature"]["dtype"] == "<f4"
                and manifest["feature"]["row_bytes"] == dim * 4 == geo.feature_row_bytes
                and manifest["feature"]["padding_node_id"] == -1, "feature geometry mismatch")
        bits = []
        for name, count in (("hot", n), ("primary", n), ("seen", g)):
            bit = Bits(root / (".compact-" + name + ".bits"), count)
            stack.callback(bit.close)
            bits.append(bit)
        hot, used, seen = bits
        hot_count = 0
        if "hot_nodes" in sources:
            h = sources["hot_nodes"]
            kind = np.dtype(h["dtype"]).kind
            require(len(h["shape"]) == 1 and kind in "iub" and (kind != "b" or h["shape"] == [n]), "invalid hot input")
            with Path(h["path"]).open("rb") as f:
                for start in range(0, h["shape"][0], chunk_rows):
                    values = lp.read_items(f, h, start, min(chunk_rows, h["shape"][0] - start))
                    for node in (np.flatnonzero(values) + start if kind == "b" else values):
                        hot_count += not hot.set(int(node))
        for gid in range(g):
            old.pause_requested()
            require(arrays.one("supernode_to_group", gid) == gid and arrays.one("group_storage_base", gid) == gid * align,
                    "group base/ID mismatch")
            require(0 <= arrays.one("group_owner", gid) < n, "invalid group owner")
            members = arrays.get("group_members", gid * size, size)
            require(len(set(int(v) for v in members)) == size, "duplicate group member")
            for local, value in enumerate(members):
                node, row = int(value), gid * align + local
                require(0 <= node < n and not hot.get(node) and arrays.one("storage_to_node", row) == node, "invalid group member")
                if gid < primary:
                    require(not used.set(node) and arrays.one("node_to_primary_row", node) == row, "primary uniqueness/inverse mismatch")
                else:
                    require(used.get(node), "replica of ungrouped node")
            for local in range(size, align):
                require(arrays.one("storage_to_node", gid * align + local) == -1, "group padding used")
        hot_start = g * align
        cold_start = ((hot_start + hot_count + align - 1) // align) * align
        cold_count = n - primary * size - hot_count
        require(cold_count >= 0 and rows == ((cold_start + cold_count + align - 1) // align) * align, "storage size mismatch")
        for is_hot, start, count, end in ((True, hot_start, hot_count, cold_start), (False, cold_start, cold_count, rows)):
            previous = -1
            for row in range(start, start + count):
                node = arrays.one("storage_to_node", row)
                require(previous < node < n and not used.get(node) and hot.get(node) == is_hot, "raw partition mismatch")
                previous = node
            for row in range(start + count, end):
                require(arrays.one("storage_to_node", row) == -1, "raw padding used")
        for node in range(n):
            row = arrays.one("node_to_primary_row", node)
            require(0 <= row < rows and arrays.one("storage_to_node", row) == node and (used.get(node) or row >= hot_start), "inverse mismatch")
        ptr, idx, feat = (sources[k] for k in ("indptr", "indices", "features"))
        require(ptr["shape"] == [n + 1] and idx["shape"] == [e] and np.dtype(ptr["dtype"]).kind in "iu"
                and np.dtype(idx["dtype"]).kind in "iu" and feat["shape"] == [n, dim]
                and feat["dtype"] == "<f4" and not feat["fortran_order"], "source metadata mismatch")
        pf = stack.enter_context(Path(ptr["path"]).open("rb"))
        inf = stack.enter_context(Path(idx["path"]).open("rb"))
        ff = stack.enter_context(Path(feat["path"]).open("rb"))
        data = arrays.files["reordered_features"]
        data.seek(arrays.info["reordered_features"]["offset"])
        for start in range(0, rows, chunk_rows):
            old.pause_requested()
            for node in arrays.get("storage_to_node", start, min(chunk_rows, rows - start)):
                require(-1 <= int(node) < n, "invalid storage row")
                expected = bytes(dim * 4)
                if node >= 0:
                    ff.seek(feat["offset"] + int(node) * dim * 4)
                    expected = ff.read(dim * 4)
                require(len(expected) == dim * 4 and data.read(dim * 4) == expected, "feature payload bit mismatch")
        original_end = rewritten_end = seen_count = max_degree = 0
        for owner in range(n + g):
            old.pause_requested()
            c, d = (int(v) for v in arrays.get("reordered_indptr", owner, 2))
            require(c == rewritten_end and c <= d <= shapes["reordered_indices"][0], "invalid rewritten pointer")
            rewritten_end = d
            if owner >= n:
                require(c == d, "nonempty supernode column")
                continue
            a, b = (int(v) for v in lp.read_items(pf, ptr, owner, 2))
            require(a == original_end and a <= b <= e and b - a <= cap, "invalid original pointer or degree cap exceeded")
            original_end = b
            max_degree = max(max_degree, b - a)
            expected = lp.read_items(inf, idx, a, b - a)
            require(not len(expected) or (expected.min() >= 0 and expected.max() < n), "invalid original neighbor")
            expanded = np.empty(b - a, dtype="<i8")
            position, last_group = 0, -1
            for start in range(c, d, chunk_rows):
                for value in arrays.get("reordered_indices", start, min(chunk_rows, d - start)):
                    value = int(value)
                    require(0 <= value < n + g, "invalid rewritten neighbor")
                    if value < n:
                        require(last_group == -1, "raw edge after group")
                        members = (value,)
                    else:
                        gid = value - n
                        require(gid > last_group and arrays.one("group_owner", gid) == owner and not seen.set(gid), "group occurrence mismatch")
                        last_group = gid
                        seen_count += 1
                        members = arrays.get("group_members", gid * size, size)
                    require(position + len(members) <= len(expanded), "too many expanded edges")
                    expanded[position:position + len(members)] = members
                    position += len(members)
            require(position == len(expected), "expanded degree mismatch")
            expanded.sort(kind="heapsort")
            require(np.array_equal(np.sort(expected, kind="heapsort"), expanded), "expanded adjacency mismatch")
        require(original_end == e and rewritten_end == shapes["reordered_indices"][0] and seen_count == g, "terminal count mismatch")
    lp.inputs_unchanged(sources)
    return dict(passed=True, schema="digit-compact-validation-v1", feature_bit_exact=True,
                expanded_adjacency_exact=True, edges=e, groups=g, storage_rows=rows,
                bitmap_scratch_bytes=scratch, maximum_degree=max_degree, degree_cap=cap,
                per_edge_sqlite=False, gpu_accessed=False, raw_ssd_accessed=False)
