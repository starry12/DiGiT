"""Versioned frequency-profiled CPU page-cache artifacts for DiGiT."""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from .artifacts import ArtifactBundle, ArtifactValidationError


CACHE_SCHEMA = "digit-cpu-cache"
CACHE_VERSION = 1
CACHE_MANIFEST = "manifest.json"
CACHE_FILES = {
    "logical_frequency": "logical_frequency.npy",
    "storage_row_frequency": "storage_row_frequency.npy",
    "cached_pages": "cached_pages.npy",
    "cached_storage_rows": "cached_storage_rows.npy",
    "hot_nodes": "hot_nodes.npy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CPUCacheSelection:
    logical_frequency: np.ndarray
    storage_row_frequency: np.ndarray
    cached_pages: np.ndarray
    cached_storage_rows: np.ndarray
    hot_nodes: np.ndarray
    statistics: Mapping


@dataclass(frozen=True)
class CPUCacheBundle:
    root: Path
    manifest: Mapping
    arrays: Mapping[str, np.ndarray]


def select_frequency_pages(
    logical_frequency,
    storage_row_frequency,
    storage_to_node,
    *,
    page_size: int,
    row_bytes: int,
    capacity_bytes: int,
) -> CPUCacheSelection:
    """Select the hottest complete storage pages using observed row accesses."""

    logical_frequency = np.asarray(logical_frequency, dtype=np.int64)
    row_frequency = np.asarray(storage_row_frequency, dtype=np.int64)
    storage_to_node = np.asarray(storage_to_node, dtype=np.int64)
    if logical_frequency.ndim != 1 or row_frequency.ndim != 1:
        raise ValueError("frequency arrays must be one-dimensional")
    if storage_to_node.shape != row_frequency.shape:
        raise ValueError("storage-row frequency and mapping shapes differ")
    if np.any(logical_frequency < 0) or np.any(row_frequency < 0):
        raise ValueError("frequency values cannot be negative")
    if page_size <= 0 or row_bytes <= 0 or page_size % row_bytes:
        raise ValueError("row bytes must divide page size")
    rows_per_page = page_size // row_bytes
    if row_frequency.size % rows_per_page:
        raise ValueError("storage rows must end on a complete page")
    capacity_pages = int(capacity_bytes) // page_size
    if capacity_pages <= 0:
        raise ValueError("CPU cache capacity is smaller than one storage page")

    page_frequency = row_frequency.reshape(-1, rows_per_page).sum(axis=1)
    page_ids = np.arange(page_frequency.size, dtype=np.int64)
    ranked = np.lexsort((page_ids, -page_frequency))
    positive = ranked[page_frequency[ranked] > 0]
    cached_pages = np.sort(positive[:capacity_pages]).astype(np.int64, copy=False)
    offsets = np.arange(rows_per_page, dtype=np.int64)
    cached_rows = (cached_pages[:, None] * rows_per_page + offsets).reshape(-1)
    mapped_nodes = storage_to_node[cached_rows] if cached_rows.size else np.empty(0, dtype=np.int64)
    hot_nodes = np.unique(mapped_nodes[mapped_nodes >= 0]).astype(np.int64, copy=False)
    total_accesses = int(row_frequency.sum())
    predicted_hits = int(row_frequency[cached_rows].sum()) if cached_rows.size else 0
    statistics = {
        "capacity_bytes": int(capacity_bytes),
        "capacity_pages": capacity_pages,
        "cached_pages": int(cached_pages.size),
        "cached_storage_rows": int(cached_rows.size),
        "hot_logical_nodes": int(hot_nodes.size),
        "profiled_row_accesses": total_accesses,
        "predicted_cpu_hits": predicted_hits,
        "predicted_cpu_hit_rate": predicted_hits / total_accesses if total_accesses else 0.0,
        "rows_per_page": rows_per_page,
    }
    return CPUCacheSelection(
        logical_frequency=logical_frequency,
        storage_row_frequency=row_frequency,
        cached_pages=cached_pages,
        cached_storage_rows=cached_rows,
        hot_nodes=hot_nodes,
        statistics=statistics,
    )


def select_packed_hot_nodes(
    logical_frequency,
    hot_nodes,
    node_to_primary_row,
    storage_to_node,
    *,
    page_size: int,
    row_bytes: int,
) -> CPUCacheSelection:
    """Bind a profiled hot-node set to a target artifact's packed raw rows."""

    logical_frequency = np.asarray(logical_frequency, dtype=np.int64)
    hot_nodes = np.asarray(hot_nodes, dtype=np.int64).reshape(-1)
    node_to_primary_row = np.asarray(node_to_primary_row, dtype=np.int64)
    storage_to_node = np.asarray(storage_to_node, dtype=np.int64)
    if logical_frequency.ndim != 1 or node_to_primary_row.shape != logical_frequency.shape:
        raise ValueError("logical frequency and primary-row mapping shapes differ")
    if np.any(logical_frequency < 0):
        raise ValueError("frequency values cannot be negative")
    if hot_nodes.size != np.unique(hot_nodes).size:
        raise ValueError("hot nodes must be unique")
    if hot_nodes.size and (hot_nodes.min() < 0 or hot_nodes.max() >= logical_frequency.size):
        raise ValueError("hot node is outside the logical ID range")
    if page_size <= 0 or row_bytes <= 0 or page_size % row_bytes:
        raise ValueError("row bytes must divide page size")
    rows_per_page = page_size // row_bytes
    cached_rows = np.sort(node_to_primary_row[hot_nodes]).astype(np.int64, copy=False)
    cached_pages = np.unique(cached_rows // rows_per_page).astype(np.int64, copy=False)
    expected_rows = (
        cached_pages[:, None] * rows_per_page
        + np.arange(rows_per_page, dtype=np.int64)
    ).reshape(-1)
    if not np.array_equal(cached_rows, expected_rows):
        raise ValueError("packed hot nodes must occupy complete storage pages")
    if cached_rows.size and (
        cached_rows[0] < 0 or cached_rows[-1] >= storage_to_node.size
    ):
        raise ValueError("packed hot storage row is out of range")
    mapped_hot = np.sort(storage_to_node[cached_rows])
    if not np.array_equal(mapped_hot, np.sort(hot_nodes)):
        raise ValueError("packed hot pages contain padding or non-hot nodes")

    storage_row_frequency = np.zeros(storage_to_node.size, dtype=np.int64)
    storage_row_frequency[node_to_primary_row] = logical_frequency
    total_accesses = int(logical_frequency.sum())
    predicted_hits = int(logical_frequency[hot_nodes].sum())
    statistics = {
        "strategy": "packed_hot_nodes",
        "capacity_bytes": int(cached_pages.size * page_size),
        "capacity_pages": int(cached_pages.size),
        "cached_pages": int(cached_pages.size),
        "cached_storage_rows": int(cached_rows.size),
        "hot_logical_nodes": int(hot_nodes.size),
        "profiled_row_accesses": total_accesses,
        "predicted_cpu_hits": predicted_hits,
        "predicted_cpu_hit_rate": predicted_hits / total_accesses if total_accesses else 0.0,
        "rows_per_page": rows_per_page,
    }
    return CPUCacheSelection(
        logical_frequency=logical_frequency.copy(),
        storage_row_frequency=storage_row_frequency,
        cached_pages=cached_pages,
        cached_storage_rows=cached_rows,
        hot_nodes=np.sort(hot_nodes),
        statistics=statistics,
    )


def save_cpu_cache_bundle(
    output_dir,
    selection: CPUCacheSelection,
    artifact: ArtifactBundle,
    *,
    profile: Mapping,
) -> CPUCacheBundle:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / CACHE_MANIFEST
    if manifest_path.exists():
        raise FileExistsError("CPU cache artifact already exists: {}".format(root))
    arrays = {
        "logical_frequency": selection.logical_frequency,
        "storage_row_frequency": selection.storage_row_frequency,
        "cached_pages": selection.cached_pages,
        "cached_storage_rows": selection.cached_storage_rows,
        "hot_nodes": selection.hot_nodes,
    }
    for name, filename in CACHE_FILES.items():
        np.save(root / filename, np.asarray(arrays[name]), allow_pickle=False)
    manifest = {
        "schema_name": CACHE_SCHEMA,
        "schema_version": CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(artifact.root.resolve()),
        "source_feature_sha256": artifact.manifest["files"]["reordered_features"]["sha256"],
        "num_nodes": artifact.num_nodes,
        "num_storage_rows": int(artifact.manifest["feature"]["num_storage_rows"]),
        "page_size": int(artifact.manifest["io"]["page_size"]),
        "row_bytes": int(artifact.manifest["feature"]["row_bytes"]),
        "profile": dict(profile),
        "statistics": dict(selection.statistics),
        "files": {},
    }
    for name, filename in CACHE_FILES.items():
        path = root / filename
        array = arrays[name]
        manifest["files"][name] = {
            "path": filename,
            "dtype": np.dtype(array.dtype).str,
            "shape": list(array.shape),
            "sha256": _sha256(path),
        }
    temporary = root / (CACHE_MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return load_cpu_cache_bundle(root, artifact)


def load_cpu_cache_bundle(root, artifact: ArtifactBundle, *, verify_checksums=True):
    root = Path(root).resolve()
    try:
        manifest = _digit_paths.json_loads((root / CACHE_MANIFEST).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("cannot read CPU cache manifest") from error
    if manifest.get("schema_name") != CACHE_SCHEMA or manifest.get("schema_version") != CACHE_VERSION:
        raise ArtifactValidationError("unsupported CPU cache artifact version")
    expected = {
        "source_artifact": str(artifact.root.resolve()),
        "source_feature_sha256": artifact.manifest["files"]["reordered_features"]["sha256"],
        "num_nodes": artifact.num_nodes,
        "num_storage_rows": int(artifact.manifest["feature"]["num_storage_rows"]),
        "page_size": int(artifact.manifest["io"]["page_size"]),
        "row_bytes": int(artifact.manifest["feature"]["row_bytes"]),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ArtifactValidationError("CPU cache manifest mismatch: {}".format(key))
    arrays = {}
    for name, filename in CACHE_FILES.items():
        entry = manifest.get("files", {}).get(name, {})
        path = root / entry.get("path", filename)
        if verify_checksums and _sha256(path) != entry.get("sha256"):
            raise ArtifactValidationError("CPU cache checksum mismatch: {}".format(name))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry.get("shape") or np.dtype(array.dtype).str != entry.get("dtype"):
            raise ArtifactValidationError("CPU cache array metadata mismatch: {}".format(name))
        arrays[name] = array

    rows = np.asarray(arrays["cached_storage_rows"])
    pages = np.asarray(arrays["cached_pages"])
    rows_per_page = expected["page_size"] // expected["row_bytes"]
    expected_rows = (
        pages[:, None] * rows_per_page + np.arange(rows_per_page, dtype=np.int64)
    ).reshape(-1)
    if rows.dtype != np.int64 or pages.dtype != np.int64 or not np.array_equal(rows, expected_rows):
        raise ArtifactValidationError("cached storage rows must contain complete sorted pages")
    if rows.size and (rows[0] < 0 or rows[-1] >= expected["num_storage_rows"]):
        raise ArtifactValidationError("cached storage row is out of range")
    return CPUCacheBundle(root=root, manifest=manifest, arrays=arrays)
