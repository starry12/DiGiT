"""Versioned deterministic logical-ID evaluation traces for DiGiT."""

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

import hashlib
import json
import os
import random
from pathlib import Path

import dgl
import numpy as np
import torch


SCHEMA_NAME = "digit-deterministic-evaluation-trace"
SCHEMA_VERSION = 1


class EvaluationTraceError(RuntimeError):
    pass


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def _numpy_int64(tensor):
    return tensor.detach().cpu().numpy().astype(np.int64, copy=False)


def block_arrays(block, layer):
    local_src, local_dst = block.edges(order="eid")
    arrays = {
        "layer{}_src_nodes".format(layer): _numpy_int64(block.srcdata[dgl.NID]),
        "layer{}_dst_nodes".format(layer): _numpy_int64(block.dstdata[dgl.NID]),
        "layer{}_local_src".format(layer): _numpy_int64(local_src),
        "layer{}_local_dst".format(layer): _numpy_int64(local_dst),
    }
    if dgl.EID in block.edata:
        arrays["layer{}_eids".format(layer)] = _numpy_int64(block.edata[dgl.EID])
    return arrays


def block_record(arrays, layer):
    prefix = "layer{}_".format(layer)
    src_nodes = arrays[prefix + "src_nodes"]
    dst_nodes = arrays[prefix + "dst_nodes"]
    local_src = arrays[prefix + "local_src"]
    local_dst = arrays[prefix + "local_dst"]
    record = {
        "layer": int(layer),
        "num_src_nodes": int(src_nodes.size),
        "num_dst_nodes": int(dst_nodes.size),
        "num_edges": int(local_src.size),
        "src_nodes_sha256": array_sha256(src_nodes),
        "dst_nodes_sha256": array_sha256(dst_nodes),
        "local_src_sha256": array_sha256(local_src),
        "local_dst_sha256": array_sha256(local_dst),
    }
    eids_key = prefix + "eids"
    if eids_key in arrays:
        record["eids_sha256"] = array_sha256(arrays[eids_key])
    return record


def batch_record(arrays, num_layers):
    return {
        "input_nodes": int(arrays["input_nodes"].size),
        "output_nodes": int(arrays["output_nodes"].size),
        "input_nodes_sha256": array_sha256(arrays["input_nodes"]),
        "output_nodes_sha256": array_sha256(arrays["output_nodes"]),
        "layers": [block_record(arrays, layer) for layer in range(num_layers)],
    }


def sample_batch(graph, sampler, output_nodes):
    input_nodes, sampled_output_nodes, blocks = sampler.sample(graph, output_nodes)
    arrays = {
        "input_nodes": _numpy_int64(input_nodes),
        "output_nodes": _numpy_int64(sampled_output_nodes),
        "num_layers": np.asarray([len(blocks)], dtype=np.int64),
    }
    for layer, block in enumerate(blocks):
        arrays.update(block_arrays(block, layer))
    return arrays


def save_batch(path, arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_batch_arrays(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def validate_batch_arrays(arrays, expected_record=None):
    required = {"input_nodes", "output_nodes", "num_layers"}
    missing = required - set(arrays)
    if missing:
        raise EvaluationTraceError(
            "evaluation batch is missing arrays: {}".format(sorted(missing))
        )
    if arrays["num_layers"].shape != (1,):
        raise EvaluationTraceError("num_layers must have shape [1]")
    num_layers = int(arrays["num_layers"][0])
    if num_layers <= 0:
        raise EvaluationTraceError("evaluation batch must contain at least one layer")
    for name in ("input_nodes", "output_nodes"):
        if arrays[name].dtype != np.dtype("int64") or arrays[name].ndim != 1:
            raise EvaluationTraceError("{} must be an int64 vector".format(name))
    for layer in range(num_layers):
        prefix = "layer{}_".format(layer)
        for suffix in ("src_nodes", "dst_nodes", "local_src", "local_dst"):
            if prefix + suffix not in arrays:
                raise EvaluationTraceError(
                    "evaluation batch is missing {}".format(prefix + suffix)
                )
        src_nodes = arrays[prefix + "src_nodes"]
        dst_nodes = arrays[prefix + "dst_nodes"]
        local_src = arrays[prefix + "local_src"]
        local_dst = arrays[prefix + "local_dst"]
        if any(value.dtype != np.dtype("int64") for value in (
            src_nodes, dst_nodes, local_src, local_dst
        )):
            raise EvaluationTraceError("evaluation block arrays must use int64")
        if local_src.shape != local_dst.shape or local_src.ndim != 1:
            raise EvaluationTraceError("local edge arrays must be matching vectors")
        if local_src.size and (
            int(local_src.min()) < 0 or int(local_src.max()) >= src_nodes.size
        ):
            raise EvaluationTraceError("local source index is outside block")
        if local_dst.size and (
            int(local_dst.min()) < 0 or int(local_dst.max()) >= dst_nodes.size
        ):
            raise EvaluationTraceError("local destination index is outside block")
        eids = arrays.get(prefix + "eids")
        if eids is not None and (eids.dtype != np.dtype("int64") or eids.shape != local_src.shape):
            raise EvaluationTraceError("block EIDs must be an int64 edge vector")
        if layer and not np.array_equal(
            arrays["layer{}_dst_nodes".format(layer - 1)], src_nodes
        ):
            raise EvaluationTraceError(
                "adjacent evaluation blocks do not share the same frontier"
            )
    if not np.array_equal(arrays["input_nodes"], arrays["layer0_src_nodes"]):
        raise EvaluationTraceError("input_nodes do not match outer block source NIDs")
    if not np.array_equal(
        arrays["output_nodes"], arrays["layer{}_dst_nodes".format(num_layers - 1)]
    ):
        raise EvaluationTraceError("output_nodes do not match inner block destination NIDs")
    record = batch_record(arrays, num_layers)
    if expected_record is not None and record != expected_record:
        raise EvaluationTraceError("evaluation batch semantic digest does not match manifest")
    return record


def reconstruct_batch(arrays, device=None):
    validate_batch_arrays(arrays)
    num_layers = int(arrays["num_layers"][0])
    blocks = []
    for layer in range(num_layers):
        prefix = "layer{}_".format(layer)
        src_nodes = torch.from_numpy(arrays[prefix + "src_nodes"].copy())
        dst_nodes = torch.from_numpy(arrays[prefix + "dst_nodes"].copy())
        local_src = torch.from_numpy(arrays[prefix + "local_src"].copy())
        local_dst = torch.from_numpy(arrays[prefix + "local_dst"].copy())
        block = dgl.create_block(
            (local_src, local_dst),
            num_src_nodes=src_nodes.numel(),
            num_dst_nodes=dst_nodes.numel(),
        )
        block.srcdata[dgl.NID] = src_nodes
        block.dstdata[dgl.NID] = dst_nodes
        eids = arrays.get(prefix + "eids")
        if eids is not None:
            block.edata[dgl.EID] = torch.from_numpy(eids.copy())
        if device is not None:
            block = block.to(device)
        blocks.append(block)
    input_nodes = torch.from_numpy(arrays["input_nodes"].copy())
    output_nodes = torch.from_numpy(arrays["output_nodes"].copy())
    if device is not None:
        input_nodes = input_nodes.to(device)
        output_nodes = output_nodes.to(device)
    return input_nodes, output_nodes, blocks


def _aggregate_semantic_sha256(records):
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_trace(
    graph,
    test_nodes,
    fanouts,
    batch_size,
    seed,
    output_dir,
    source=None,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        trace = EvaluationTrace(output_dir, verify_files=True)
        trace.validate_contract(
            graph.num_nodes(), graph.num_edges(), fanouts, batch_size, test_nodes
        )
        return trace

    test_nodes = np.asarray(test_nodes, dtype=np.int64)
    if test_nodes.ndim != 1 or test_nodes.size == 0:
        raise EvaluationTraceError("test_nodes must be a non-empty int64 vector")
    fanouts = [int(value) for value in fanouts]
    if not fanouts or any(value <= 0 for value in fanouts):
        raise EvaluationTraceError("fanouts must be positive")
    if batch_size <= 0:
        raise EvaluationTraceError("batch_size must be positive")
    _seed_all(seed)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    records = []
    num_batches = int(math_ceil_div(test_nodes.size, batch_size))
    for batch_index in range(num_batches):
        begin = batch_index * batch_size
        end = min(begin + batch_size, test_nodes.size)
        seeds = torch.from_numpy(test_nodes[begin:end].copy())
        arrays = sample_batch(graph, sampler, seeds)
        record = validate_batch_arrays(arrays)
        filename = "batch_{:06d}.npz".format(batch_index)
        path = output_dir / filename
        if path.exists():
            existing = load_batch_arrays(path)
            validate_batch_arrays(existing, record)
        else:
            save_batch(path, arrays)
        record.update({
            "batch": batch_index,
            "path": filename,
            "sha256": sha256_file(path),
        })
        records.append(record)

    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "graph": {
            "num_nodes": int(graph.num_nodes()),
            "num_edges": int(graph.num_edges()),
        },
        "sampling": {
            "policy": "standard_uniform_logical",
            "fanouts": fanouts,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "num_batches": num_batches,
            "num_test_nodes": int(test_nodes.size),
            "test_nodes_sha256": array_sha256(test_nodes),
        },
        "runtime": {
            "dgl": dgl.__version__,
            "torch": torch.__version__,
        },
        "source": source or {},
        "aggregate_semantic_sha256": _aggregate_semantic_sha256(records),
        "batches": records,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return EvaluationTrace(output_dir, verify_files=True)


def math_ceil_div(numerator, denominator):
    return (int(numerator) + int(denominator) - 1) // int(denominator)


class EvaluationTrace:
    def __init__(self, root, verify_files=True):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise EvaluationTraceError(
                "evaluation trace manifest is missing: {}".format(self.manifest_path)
            )
        self.manifest = _digit_paths.json_loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_name") != SCHEMA_NAME:
            raise EvaluationTraceError("unexpected evaluation trace schema name")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise EvaluationTraceError("unsupported evaluation trace schema version")
        records = self.manifest.get("batches", [])
        sampling = self.manifest.get("sampling", {})
        if len(records) != sampling.get("num_batches"):
            raise EvaluationTraceError("evaluation trace batch count mismatch")
        if _aggregate_semantic_sha256(records) != self.manifest.get(
            "aggregate_semantic_sha256"
        ):
            raise EvaluationTraceError("evaluation trace aggregate digest mismatch")
        self.verify_files = bool(verify_files)
        for expected_index, record in enumerate(records):
            if record.get("batch") != expected_index:
                raise EvaluationTraceError("evaluation trace batch order mismatch")
            path = self.root / record["path"]
            if not path.is_file():
                raise EvaluationTraceError("evaluation trace batch is missing: {}".format(path))

    @property
    def semantic_sha256(self):
        return self.manifest["aggregate_semantic_sha256"]

    @property
    def manifest_sha256(self):
        return sha256_file(self.manifest_path)

    def validate_contract(self, num_nodes, num_edges, fanouts, batch_size, test_nodes):
        graph = self.manifest["graph"]
        sampling = self.manifest["sampling"]
        if graph != {"num_nodes": int(num_nodes), "num_edges": int(num_edges)}:
            raise EvaluationTraceError("evaluation trace graph identity mismatch")
        if sampling["fanouts"] != [int(value) for value in fanouts]:
            raise EvaluationTraceError("evaluation trace fanouts mismatch")
        if sampling["batch_size"] != int(batch_size):
            raise EvaluationTraceError("evaluation trace batch size mismatch")
        test_nodes = np.asarray(test_nodes, dtype=np.int64)
        if sampling["num_test_nodes"] != int(test_nodes.size):
            raise EvaluationTraceError("evaluation trace test-node count mismatch")
        if sampling["test_nodes_sha256"] != array_sha256(test_nodes):
            raise EvaluationTraceError("evaluation trace test-node digest mismatch")

    def iter_batches(self, device=None):
        for record in self.manifest["batches"]:
            path = self.root / record["path"]
            if self.verify_files and sha256_file(path) != record["sha256"]:
                raise EvaluationTraceError(
                    "evaluation trace batch checksum mismatch: {}".format(path)
                )
            arrays = load_batch_arrays(path)
            expected = {
                key: value for key, value in record.items()
                if key not in ("batch", "path", "sha256")
            }
            validate_batch_arrays(arrays, expected)
            yield reconstruct_batch(arrays, device=device)
