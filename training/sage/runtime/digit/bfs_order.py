"""Versioned deterministic BFS training-seed order artifacts."""

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
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .artifacts import ArtifactValidationError


SCHEMA_NAME = "digit-bfs-seed-order"
SCHEMA_VERSION = 1
MANIFEST = "manifest.json"
ORDER_FILE = "train_order.npy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class BFSOrderBundle:
    root: Path
    manifest: Mapping
    order: np.ndarray


def shuffle_bfs_batch_blocks(order, batch_size: int, seed: int, epoch: int = 0):
    """Shuffle complete BFS minibatch blocks without changing nodes within them.

    A short final block remains last. Moving it between full blocks would make
    the fixed-size DGL batching boundary combine it with the following block,
    which would destroy both original BFS blocks.
    """

    values = np.asarray(order)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("order must be a one-dimensional integer array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    complete_blocks = int(values.size // batch_size)
    full_end = complete_blocks * batch_size
    permutation = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch)])
    ).permutation(complete_blocks)

    shuffled = np.empty(values.shape, dtype=values.dtype)
    for output_block, input_block in enumerate(permutation):
        output_start = output_block * batch_size
        input_start = int(input_block) * batch_size
        shuffled[output_start : output_start + batch_size] = values[
            input_start : input_start + batch_size
        ]
    shuffled[full_end:] = values[full_end:]
    metadata = {
        "policy": "bfs_batch_block_shuffle",
        "seed": int(seed),
        "epoch": int(epoch),
        "batch_size": int(batch_size),
        "complete_blocks_shuffled": complete_blocks,
        "tail_nodes_kept_last": int(values.size - full_end),
        "block_permutation_sha256": _array_sha256(
            permutation.astype(np.int64, copy=False)
        ),
        "ordered_nodes_sha256": _array_sha256(shuffled),
    }
    return shuffled, metadata


def _validate_inputs(edge_index, train_nodes, num_nodes):
    edges = np.asarray(edge_index)
    train = np.asarray(train_nodes)
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index must be an integer array with shape [2, E]")
    if train.ndim != 1 or not np.issubdtype(train.dtype, np.integer):
        raise ValueError("train_nodes must be a one-dimensional integer array")
    train = train.astype(np.int64, copy=False)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if train.size != np.unique(train).size:
        raise ValueError("train_nodes must be unique")
    if train.size and (train.min() < 0 or train.max() >= num_nodes):
        raise ValueError("training node is outside the graph")
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("edge endpoint is outside the graph")
    return edges, np.sort(train)


def build_bfs_train_order(edge_index, train_nodes, num_nodes: int):
    """Return canonical BFS order and structural statistics.

    BFS runs on the undirected training-induced graph. Components start at the
    highest-degree unvisited node (smallest logical ID breaks ties), and each
    adjacency list is traversed in ascending logical-ID order.
    """

    edges, train = _validate_inputs(edge_index, train_nodes, num_nodes)
    train_count = int(train.size)
    global_to_local = np.full(num_nodes, -1, dtype=np.int64)
    global_to_local[train] = np.arange(train_count, dtype=np.int64)
    src = global_to_local[np.asarray(edges[0], dtype=np.int64)]
    dst = global_to_local[np.asarray(edges[1], dtype=np.int64)]
    keep = (src >= 0) & (dst >= 0) & (src != dst)
    src = src[keep]
    dst = dst[keep]
    undirected_src = np.concatenate((src, dst))
    undirected_dst = np.concatenate((dst, src))
    if undirected_src.size:
        sort_order = np.lexsort((undirected_dst, undirected_src))
        undirected_src = undirected_src[sort_order]
        undirected_dst = undirected_dst[sort_order]
    degree = np.bincount(undirected_src, minlength=train_count).astype(np.int64)
    indptr = np.empty(train_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])

    # lexsort uses the final key as primary: descending degree, then global ID.
    root_priority = np.lexsort((train, -degree))
    visited = np.zeros(train_count, dtype=bool)
    local_order = np.empty(train_count, dtype=np.int64)
    cursor = 0
    components = 0
    max_frontier = 0
    queue = deque()
    for root in root_priority:
        root = int(root)
        if visited[root]:
            continue
        components += 1
        visited[root] = True
        queue.append(root)
        while queue:
            max_frontier = max(max_frontier, len(queue))
            node = queue.popleft()
            local_order[cursor] = node
            cursor += 1
            start, end = int(indptr[node]), int(indptr[node + 1])
            for neighbor in undirected_dst[start:end]:
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    if cursor != train_count:
        raise RuntimeError("BFS did not emit every training node")
    order = train[local_order]
    statistics = {
        "num_components": components,
        "max_queue_size": max_frontier,
        "directed_training_edges": int(keep.sum()),
        "undirected_adjacency_entries": int(undirected_src.size),
        "isolated_training_nodes": int(np.count_nonzero(degree == 0)),
        "max_training_degree": int(degree.max()) if degree.size else 0,
    }
    return order, statistics


def same_batch_edge_rate(edge_index, train_nodes, order, batch_size: int) -> Mapping:
    """Measure the fraction of train-induced edges whose endpoints share a batch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    edges = np.asarray(edge_index)
    train = np.asarray(train_nodes, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)
    num_nodes = int(max(edges.max(initial=-1), train.max(initial=-1)) + 1)
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train] = True
    keep = train_mask[edges[0]] & train_mask[edges[1]]
    position = np.full(num_nodes, -1, dtype=np.int64)
    position[order] = np.arange(order.size, dtype=np.int64)
    same = position[edges[0, keep]] // batch_size == position[edges[1, keep]] // batch_size
    count = int(keep.sum())
    return {
        "training_edges": count,
        "same_batch_edges": int(same.sum()),
        "same_batch_edge_rate": float(same.mean()) if count else 0.0,
        "batch_size": int(batch_size),
    }


def save_bfs_order_bundle(
    output_dir,
    order,
    *,
    train_nodes,
    num_nodes: int,
    dataset_name: str,
    dataset_size: str,
    edge_index_path,
    algorithm_statistics: Mapping,
    locality: Mapping,
) -> BFSOrderBundle:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST
    order_path = root / ORDER_FILE
    if manifest_path.exists() or order_path.exists():
        raise FileExistsError("BFS order artifact already exists: {}".format(root))
    order = np.asarray(order, dtype=np.int64)
    train = np.sort(np.asarray(train_nodes, dtype=np.int64))
    if order.size != train.size or not np.array_equal(np.sort(order), train):
        raise ValueError("BFS order must be an exact permutation of training nodes")
    np.save(order_path, order, allow_pickle=False)
    edge_path = Path(edge_index_path).resolve()
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "size": dataset_size,
            "num_nodes": int(num_nodes),
            "num_train_nodes": int(train.size),
        },
        "algorithm": {
            "name": "deterministic_training_induced_undirected_bfs",
            "component_root": "descending_training_degree_then_logical_id",
            "neighbor_order": "ascending_logical_id",
            "statistics": dict(algorithm_statistics),
        },
        "source": {
            "edge_index": str(edge_path),
            "edge_index_sha256": _sha256(edge_path),
            "train_nodes_sha256": _array_sha256(train),
        },
        "locality": dict(locality),
        "files": {
            "train_order": {
                "path": ORDER_FILE,
                "dtype": np.dtype(order.dtype).str,
                "shape": list(order.shape),
                "sha256": _sha256(order_path),
            }
        },
    }
    temporary = root / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return load_bfs_order_bundle(root, expected_train_nodes=train, num_nodes=num_nodes)


def load_bfs_order_bundle(
    root,
    *,
    expected_train_nodes: Optional[np.ndarray] = None,
    num_nodes: Optional[int] = None,
    verify_checksum: bool = True,
    verify_source: bool = True,
) -> BFSOrderBundle:
    root = Path(root).resolve()
    try:
        manifest = _digit_paths.json_loads((root / MANIFEST).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("cannot read BFS order manifest") from error
    if manifest.get("schema_name") != SCHEMA_NAME or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported BFS order artifact version")
    entry = manifest.get("files", {}).get("train_order", {})
    path = root / entry.get("path", ORDER_FILE)
    if verify_checksum and _sha256(path) != entry.get("sha256"):
        raise ArtifactValidationError("BFS order checksum mismatch")
    order = np.load(path, mmap_mode="r", allow_pickle=False)
    if order.dtype != np.int64 or order.ndim != 1:
        raise ArtifactValidationError("BFS order must be one-dimensional int64")
    if list(order.shape) != entry.get("shape") or np.dtype(order.dtype).str != entry.get("dtype"):
        raise ArtifactValidationError("BFS order metadata mismatch")
    dataset = manifest.get("dataset", {})
    if order.size != dataset.get("num_train_nodes") or np.unique(order).size != order.size:
        raise ArtifactValidationError("BFS order is not a unique training permutation")
    expected_num_nodes = int(dataset.get("num_nodes", -1))
    if num_nodes is not None and expected_num_nodes != int(num_nodes):
        raise ArtifactValidationError("BFS order graph node count mismatch")
    if order.size and (order.min() < 0 or order.max() >= expected_num_nodes):
        raise ArtifactValidationError("BFS order contains an invalid node ID")
    if expected_train_nodes is not None:
        train = np.sort(np.asarray(expected_train_nodes, dtype=np.int64))
        if _array_sha256(train) != manifest.get("source", {}).get("train_nodes_sha256"):
            raise ArtifactValidationError("BFS order training split checksum mismatch")
        if not np.array_equal(np.sort(order), train):
            raise ArtifactValidationError("BFS order differs from the current training split")
    if verify_source:
        source = manifest.get("source", {})
        edge_path = Path(source.get("edge_index", ""))
        if not edge_path.is_file() or _sha256(edge_path) != source.get("edge_index_sha256"):
            raise ArtifactValidationError("BFS order source graph checksum mismatch")
    return BFSOrderBundle(root=root, manifest=manifest, order=order)
