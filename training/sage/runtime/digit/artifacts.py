"""Versioned on-disk artifact format for DiGiT graph reorganization.

The format deliberately separates logical node IDs, group/supernode IDs, and
SSD storage-row IDs.  The manifest is written last, so its presence indicates
that all required NumPy arrays were successfully emitted.
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

import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .io_geometry import IOGeometry, GeometryValidationError


SCHEMA_NAME = "digit-artifact-bundle"
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
MANIFEST_FILENAME = "manifest.json"

ARRAY_FILENAMES = {
    "group_members": "group_members.npy",
    "group_owner": "group_owner.npy",
    "group_storage_base": "group_storage_base.npy",
    "storage_to_node": "storage_to_node.npy",
    "node_to_primary_row": "node_to_primary_row.npy",
    "supernode_to_group": "supernode_to_group.npy",
    "reordered_indptr": "reordered_indptr.npy",
    "reordered_indices": "reordered_indices.npy",
    "reordered_features": "reordered_features.npy",
}

INTEGER_ARRAYS = {
    "group_members",
    "group_owner",
    "group_storage_base",
    "storage_to_node",
    "node_to_primary_row",
    "supernode_to_group",
    "reordered_indptr",
    "reordered_indices",
}


class ArtifactValidationError(ValueError):
    """Raised when a DiGiT artifact bundle violates its versioned contract."""


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ArtifactValidationError(
            "{} is missing required keys: {}".format(context, ", ".join(missing))
        )


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactValidationError("{} must be an integer".format(name))
    return int(value)


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    _require_keys(
        manifest,
        [
            "schema_name",
            "schema_version",
            "dataset",
            "feature",
            "grouping",
            "io",
            "graph",
            "files",
        ],
        "manifest",
    )
    if manifest["schema_name"] != SCHEMA_NAME:
        raise ArtifactValidationError(
            "unsupported schema_name {!r}".format(manifest["schema_name"])
        )
    if manifest["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise ArtifactValidationError(
            "unsupported schema_version {}; expected one of {}".format(
                manifest["schema_version"], SUPPORTED_SCHEMA_VERSIONS
            )
        )
    if manifest["schema_version"] == 2:
        _require_keys(
            manifest, ["io_geometry", "io_geometry_sha256"], "manifest"
        )

    _require_keys(manifest["dataset"], ["name", "size", "num_nodes", "num_edges"], "dataset")
    _require_keys(
        manifest["feature"],
        ["dim", "dtype", "row_bytes", "num_storage_rows"],
        "feature",
    )
    _require_keys(
        manifest["grouping"],
        [
            "group_size",
            "replication_ratio",
            "num_groups",
            "num_primary_groups",
            "num_replica_groups",
            "supernode_id_start",
        ],
        "grouping",
    )
    _require_keys(manifest["io"], ["page_size", "alignment_rows"], "io")
    _require_keys(
        manifest["graph"],
        ["format", "num_logical_nodes", "num_supernodes", "num_graph_nodes"],
        "graph",
    )

    files = manifest["files"]
    _require_keys(files, list(ARRAY_FILENAMES), "files")
    for name, entry in files.items():
        if name not in ARRAY_FILENAMES:
            continue
        _require_keys(entry, ["path", "dtype", "shape", "sha256"], "files.{}".format(name))
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactValidationError(
                "files.{}.path must stay inside the bundle".format(name)
            )


def _array_metadata(name: str, array: np.ndarray, path: str, sha256: str = "") -> Dict[str, Any]:
    return {
        "path": path,
        "dtype": np.dtype(array.dtype).str,
        "shape": list(array.shape),
        "sha256": sha256,
    }


def _build_manifest(
    arrays: Mapping[str, np.ndarray],
    dataset_name: str,
    dataset_size: str,
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    group_size: int,
    replication_ratio: float,
    num_primary_groups: int,
    page_size: int,
    source: Optional[Mapping[str, Any]],
    metadata: Optional[Mapping[str, Any]],
    minimum_transfer_bytes: Optional[int] = None,
    target_request_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    num_groups = int(arrays["group_members"].shape[0])
    num_storage_rows = int(arrays["storage_to_node"].shape[0])
    feature_dtype = np.dtype(arrays["reordered_features"].dtype)
    row_bytes = feature_dtype.itemsize * feature_dim
    supernode_id_start = num_nodes
    minimum_transfer = (
        min(4096, int(page_size))
        if minimum_transfer_bytes is None
        else int(minimum_transfer_bytes)
    )
    try:
        geometry = IOGeometry.create(
            feature_row_bytes=int(row_bytes),
            group_size=int(group_size),
            minimum_transfer_bytes=minimum_transfer,
            target_request_bytes=(
                int(page_size) if target_request_bytes is None
                else int(target_request_bytes)
            ),
        )
    except GeometryValidationError as error:
        raise ArtifactValidationError(
            "invalid artifact I/O geometry: {}".format(error)
        ) from error
    if geometry.cache_slot_bytes != int(page_size):
        raise ArtifactValidationError(
            "page_size does not match the rounded group cache-slot geometry"
        )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "size": dataset_size,
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
        },
        "feature": {
            "dim": int(feature_dim),
            "dtype": feature_dtype.str,
            "row_bytes": int(row_bytes),
            "num_storage_rows": num_storage_rows,
            "padding_node_id": -1,
        },
        "grouping": {
            "group_size": int(group_size),
            "replication_ratio": float(replication_ratio),
            "num_groups": num_groups,
            "num_primary_groups": int(num_primary_groups),
            "num_replica_groups": num_groups - int(num_primary_groups),
            "supernode_id_start": supernode_id_start,
        },
        "io": {
            "page_size": int(page_size),
            "alignment_rows": max(1, int(page_size) // int(row_bytes)),
        },
        "io_geometry": geometry.to_dict(),
        "io_geometry_sha256": geometry.semantic_sha256(),
        "graph": {
            "format": "csc",
            "num_logical_nodes": int(num_nodes),
            "num_supernodes": num_groups,
            "num_graph_nodes": int(num_nodes) + num_groups,
        },
        "files": {
            name: _array_metadata(name, arrays[name], filename)
            for name, filename in ARRAY_FILENAMES.items()
        },
        "source": dict(source or {}),
        "metadata": dict(metadata or {}),
    }


def validate_artifact_bundle(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    """Validate manifest metadata and cross-array ID invariants."""

    _validate_manifest_structure(manifest)
    _require_keys(arrays, list(ARRAY_FILENAMES), "arrays")

    for name in ARRAY_FILENAMES:
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise ArtifactValidationError("arrays.{} must be a NumPy array".format(name))
        if array.dtype.hasobject:
            raise ArtifactValidationError("arrays.{} cannot use object dtype".format(name))
        if name in INTEGER_ARRAYS and not np.issubdtype(array.dtype, np.integer):
            raise ArtifactValidationError("arrays.{} must use an integer dtype".format(name))
        entry = manifest["files"][name]
        if list(array.shape) != list(entry["shape"]):
            raise ArtifactValidationError("arrays.{} shape does not match manifest".format(name))
        if np.dtype(array.dtype).str != entry["dtype"]:
            raise ArtifactValidationError("arrays.{} dtype does not match manifest".format(name))

    dataset = manifest["dataset"]
    feature = manifest["feature"]
    grouping = manifest["grouping"]
    graph = manifest["graph"]
    io = manifest["io"]

    if manifest["schema_version"] == 2:
        try:
            geometry = IOGeometry.from_dict(manifest["io_geometry"])
        except GeometryValidationError as error:
            raise ArtifactValidationError(
                "invalid io_geometry: {}".format(error)
            ) from error
        if manifest["io_geometry_sha256"] != geometry.semantic_sha256():
            raise ArtifactValidationError("io_geometry_sha256 is inconsistent")
    else:
        geometry = None

    num_nodes = _as_int(dataset["num_nodes"], "dataset.num_nodes")
    num_edges = _as_int(dataset["num_edges"], "dataset.num_edges")
    feature_dim = _as_int(feature["dim"], "feature.dim")
    num_storage_rows = _as_int(feature["num_storage_rows"], "feature.num_storage_rows")
    group_size = _as_int(grouping["group_size"], "grouping.group_size")
    num_groups = _as_int(grouping["num_groups"], "grouping.num_groups")
    num_primary = _as_int(grouping["num_primary_groups"], "grouping.num_primary_groups")
    num_replica = _as_int(grouping["num_replica_groups"], "grouping.num_replica_groups")
    supernode_start = _as_int(grouping["supernode_id_start"], "grouping.supernode_id_start")
    page_size = _as_int(io["page_size"], "io.page_size")
    alignment_rows = _as_int(io["alignment_rows"], "io.alignment_rows")

    if min(num_nodes, feature_dim, num_storage_rows, group_size, page_size, alignment_rows) <= 0:
        raise ArtifactValidationError("node, feature, storage, group, and I/O sizes must be positive")
    if num_edges < 0 or num_groups < 0:
        raise ArtifactValidationError("edge and group counts cannot be negative")
    if not 0 <= num_primary <= num_groups:
        raise ArtifactValidationError("num_primary_groups is out of range")
    if num_replica != num_groups - num_primary:
        raise ArtifactValidationError("num_replica_groups is inconsistent")
    if supernode_start != num_nodes:
        raise ArtifactValidationError("v1 supernode IDs must begin at num_nodes")
    if graph["format"] != "csc":
        raise ArtifactValidationError("v1 graph format must be csc")
    if int(graph["num_logical_nodes"]) != num_nodes:
        raise ArtifactValidationError("graph.num_logical_nodes is inconsistent")
    if int(graph["num_supernodes"]) != num_groups:
        raise ArtifactValidationError("graph.num_supernodes is inconsistent")
    if int(graph["num_graph_nodes"]) != num_nodes + num_groups:
        raise ArtifactValidationError("graph.num_graph_nodes is inconsistent")

    group_members = arrays["group_members"]
    group_owner = arrays["group_owner"]
    group_storage_base = arrays["group_storage_base"]
    storage_to_node = arrays["storage_to_node"]
    node_to_primary = arrays["node_to_primary_row"]
    supernode_to_group = arrays["supernode_to_group"]
    indptr = arrays["reordered_indptr"]
    indices = arrays["reordered_indices"]
    features = arrays["reordered_features"]

    if group_members.shape != (num_groups, group_size):
        raise ArtifactValidationError("group_members must have shape [num_groups, group_size]")
    if group_owner.shape != (num_groups,):
        raise ArtifactValidationError("group_owner must have shape [num_groups]")
    if group_storage_base.shape != (num_groups,):
        raise ArtifactValidationError("group_storage_base must have shape [num_groups]")
    if storage_to_node.shape != (num_storage_rows,):
        raise ArtifactValidationError("storage_to_node must have shape [num_storage_rows]")
    if node_to_primary.shape != (num_nodes,):
        raise ArtifactValidationError("node_to_primary_row must have shape [num_nodes]")
    if supernode_to_group.shape != (num_groups,):
        raise ArtifactValidationError("supernode_to_group must have shape [num_groups]")
    if features.shape != (num_storage_rows, feature_dim):
        raise ArtifactValidationError("reordered_features shape is inconsistent")

    row_bytes = features.dtype.itemsize * feature_dim
    if np.dtype(feature["dtype"]) != features.dtype:
        raise ArtifactValidationError("feature.dtype is inconsistent")
    if int(feature["row_bytes"]) != row_bytes:
        raise ArtifactValidationError("feature.row_bytes is inconsistent")
    if page_size % row_bytes != 0:
        raise ArtifactValidationError("page_size must be a multiple of one feature row")
    if alignment_rows != page_size // row_bytes:
        raise ArtifactValidationError("io.alignment_rows is inconsistent")
    if geometry is not None:
        if geometry.feature_row_bytes != row_bytes:
            raise ArtifactValidationError("io_geometry feature row is inconsistent")
        if geometry.group_size != group_size:
            raise ArtifactValidationError("io_geometry group size is inconsistent")
        if geometry.cache_slot_bytes != page_size:
            raise ArtifactValidationError("io_geometry cache slot is inconsistent")
        if geometry.rows_per_slot != alignment_rows:
            raise ArtifactValidationError("io_geometry alignment rows are inconsistent")

    if num_groups:
        if group_members.min() < 0 or group_members.max() >= num_nodes:
            raise ArtifactValidationError("group_members contains an invalid logical node ID")
        if group_owner.min() < 0 or group_owner.max() >= num_nodes:
            raise ArtifactValidationError("group_owner contains an invalid logical node ID")
        if group_storage_base.min() < 0:
            raise ArtifactValidationError("group_storage_base cannot be negative")
        group_rows = group_storage_base[:, None] + np.arange(group_size, dtype=np.int64)[None, :]
        if group_rows.max() >= num_storage_rows:
            raise ArtifactValidationError("a group extends beyond the feature storage")
        if np.unique(group_rows).size != group_rows.size:
            raise ArtifactValidationError("group storage-row ranges overlap")
        if not np.array_equal(storage_to_node[group_rows], group_members):
            raise ArtifactValidationError("group storage rows do not match group_members")
        if np.any(group_storage_base % alignment_rows != 0):
            raise ArtifactValidationError("group storage bases are not page aligned")
        if geometry is not None:
            group_slots = (
                group_storage_base[:, None]
                + np.arange(alignment_rows, dtype=np.int64)[None, :]
            )
            if group_slots.max() >= num_storage_rows:
                raise ArtifactValidationError("a group cache slot extends beyond storage")
            if np.unique(group_slots).size != group_slots.size:
                raise ArtifactValidationError("group cache-slot ranges overlap")
            if alignment_rows > group_size and np.any(
                storage_to_node[group_slots[:, group_size:]] != -1
            ):
                raise ArtifactValidationError("group cache-slot padding must remain unused")

    if storage_to_node.size:
        if storage_to_node.min() < -1 or storage_to_node.max() >= num_nodes:
            raise ArtifactValidationError("storage_to_node contains an invalid ID")
    if node_to_primary.min() < 0 or node_to_primary.max() >= num_storage_rows:
        raise ArtifactValidationError("node_to_primary_row contains an invalid storage row")
    if not np.array_equal(storage_to_node[node_to_primary], np.arange(num_nodes)):
        raise ArtifactValidationError("node_to_primary_row does not map back to each logical node")

    if num_primary:
        primary_members = group_members[:num_primary].reshape(-1)
        if np.unique(primary_members).size != primary_members.size:
            raise ArtifactValidationError("primary groups must be disjoint")
        primary_rows = (
            group_storage_base[:num_primary, None]
            + np.arange(group_size, dtype=np.int64)[None, :]
        ).reshape(-1)
        if not np.array_equal(node_to_primary[primary_members], primary_rows):
            raise ArtifactValidationError(
                "primary grouped nodes must use their primary group storage rows"
            )

    expected_group_ids = np.arange(num_groups, dtype=supernode_to_group.dtype)
    if not np.array_equal(np.sort(supernode_to_group), expected_group_ids):
        raise ArtifactValidationError("supernode_to_group must be a permutation of group IDs")

    replication_ratio = float(grouping["replication_ratio"])
    if replication_ratio < 0:
        raise ArtifactValidationError("replication_ratio cannot be negative")
    if num_replica * group_size > int(np.floor(replication_ratio * num_nodes)):
        raise ArtifactValidationError("replica groups exceed the replication-row budget")

    num_graph_nodes = num_nodes + num_groups
    if indptr.shape != (num_graph_nodes + 1,):
        raise ArtifactValidationError("reordered_indptr must have num_graph_nodes + 1 entries")
    if indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
        raise ArtifactValidationError("reordered_indptr must start at zero and be monotonic")
    if int(indptr[-1]) != indices.size:
        raise ArtifactValidationError("reordered_indptr[-1] must equal reordered_indices size")
    if indices.size and (indices.min() < 0 or indices.max() >= num_graph_nodes):
        raise ArtifactValidationError("reordered_indices contains an invalid graph node ID")


@dataclass(frozen=True)
class ArtifactBundle:
    root: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    @property
    def num_nodes(self) -> int:
        return int(self.manifest["dataset"]["num_nodes"])

    @property
    def num_groups(self) -> int:
        return int(self.manifest["grouping"]["num_groups"])

    @property
    def group_size(self) -> int:
        return int(self.manifest["grouping"]["group_size"])

    @property
    def supernode_id_start(self) -> int:
        return int(self.manifest["grouping"]["supernode_id_start"])

    @property
    def io_geometry(self) -> IOGeometry:
        if int(self.manifest["schema_version"]) == 2:
            return IOGeometry.from_dict(self.manifest["io_geometry"])
        return IOGeometry.from_artifact_manifest(
            self.manifest,
            minimum_transfer_bytes=min(
                4096, int(self.manifest["io"]["page_size"])
            ),
            target_request_bytes=int(self.manifest["io"]["page_size"]),
        )

    def primary_rows_for_nodes(self, logical_node_ids: np.ndarray) -> np.ndarray:
        node_ids = np.asarray(logical_node_ids, dtype=np.int64)
        if node_ids.size and (node_ids.min() < 0 or node_ids.max() >= self.num_nodes):
            raise IndexError("logical node ID is out of range")
        return np.asarray(self.arrays["node_to_primary_row"][node_ids])

    def group_ids_for_supernodes(self, supernode_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(supernode_ids, dtype=np.int64)
        local_ids = ids - self.supernode_id_start
        if local_ids.size and (local_ids.min() < 0 or local_ids.max() >= self.num_groups):
            raise IndexError("supernode ID is out of range")
        return np.asarray(self.arrays["supernode_to_group"][local_ids])

    def members_for_supernodes(self, supernode_ids: np.ndarray) -> np.ndarray:
        group_ids = self.group_ids_for_supernodes(supernode_ids)
        return np.asarray(self.arrays["group_members"][group_ids])

    def storage_rows_for_groups(self, group_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(group_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_groups):
            raise IndexError("group ID is out of range")
        bases = np.asarray(self.arrays["group_storage_base"][ids], dtype=np.int64)
        return bases[..., None] + np.arange(self.group_size, dtype=np.int64)


def save_artifact_bundle(
    output_dir: os.PathLike,
    arrays: Mapping[str, np.ndarray],
    *,
    dataset_name: str,
    dataset_size: str,
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    group_size: int,
    replication_ratio: float,
    num_primary_groups: int,
    page_size: int,
    source: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    minimum_transfer_bytes: Optional[int] = None,
    target_request_bytes: Optional[int] = None,
) -> ArtifactBundle:
    """Validate and persist a complete v2 artifact bundle."""

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / MANIFEST_FILENAME).exists():
        raise FileExistsError("artifact manifest already exists: {}".format(root))
    _require_keys(arrays, list(ARRAY_FILENAMES), "arrays")
    normalized = {name: np.asarray(arrays[name]) for name in ARRAY_FILENAMES}

    for name, filename in ARRAY_FILENAMES.items():
        path = root / filename
        if path.exists():
            raise FileExistsError("artifact array already exists: {}".format(path))
        np.save(path, normalized[name], allow_pickle=False)

    return finalize_artifact_bundle(
        root,
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        num_nodes=num_nodes,
        num_edges=num_edges,
        feature_dim=feature_dim,
        group_size=group_size,
        replication_ratio=replication_ratio,
        num_primary_groups=num_primary_groups,
        page_size=page_size,
        source=source,
        metadata=metadata,
        minimum_transfer_bytes=minimum_transfer_bytes,
        target_request_bytes=target_request_bytes,
    )


def finalize_artifact_bundle(
    output_dir: os.PathLike,
    *,
    dataset_name: str,
    dataset_size: str,
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    group_size: int,
    replication_ratio: float,
    num_primary_groups: int,
    page_size: int,
    source: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    minimum_transfer_bytes: Optional[int] = None,
    target_request_bytes: Optional[int] = None,
) -> ArtifactBundle:
    """Validate prewritten arrays and publish their manifest atomically.

    This entry point lets preprocessing stream ``reordered_features.npy``
    directly into its final location. The manifest remains absent until every
    required array passes validation and receives a checksum.
    """

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / MANIFEST_FILENAME).exists():
        raise FileExistsError("artifact manifest already exists: {}".format(root))

    arrays: Dict[str, np.ndarray] = {}
    for name, filename in ARRAY_FILENAMES.items():
        path = root / filename
        if not path.is_file():
            raise ArtifactValidationError("artifact file is missing: {}".format(path))
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)

    manifest = _build_manifest(
        arrays,
        dataset_name,
        dataset_size,
        num_nodes,
        num_edges,
        feature_dim,
        group_size,
        replication_ratio,
        num_primary_groups,
        page_size,
        source,
        metadata,
        minimum_transfer_bytes,
        target_request_bytes,
    )
    validate_artifact_bundle(manifest, arrays)

    for name, filename in ARRAY_FILENAMES.items():
        path = root / filename
        manifest["files"][name]["sha256"] = _sha256(path)

    temporary_manifest = root / (MANIFEST_FILENAME + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(root / MANIFEST_FILENAME)
    return load_artifact_bundle(root, mmap_mode="r", verify_checksums=True)


_FAST_VALIDATION_CACHE = set()

def load_artifact_bundle(
    path: os.PathLike,
    mmap_mode: Optional[str] = "r",
    verify_checksums: bool = True,
    validation_mode: Optional[str] = None,
    validation_chunk_rows: int = 4096,
    validation_scratch_dir: Optional[str] = None,
) -> ArtifactBundle:
    """Load a bundle and validate its schema, files, checksums, and mappings."""

    requested = Path(path).resolve()
    manifest_path = requested if requested.name == MANIFEST_FILENAME else requested / MANIFEST_FILENAME
    root = manifest_path.parent
    if validation_mode is None:
        validation_mode = os.environ.get("DIGIT_VALIDATION_PROFILE", "bounded")
    if validation_mode not in ("bounded", "legacy", "fast"):
        raise ArtifactValidationError("unknown validation mode")
    if validation_mode in ("bounded", "fast") and (mmap_mode != "r" or manifest_path.stat().st_size > 1024**2):
        raise ArtifactValidationError("bounded loading requires mmap_mode='r' and manifest <=1 MiB")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _digit_paths.json_loads(manifest_bytes)
    _validate_manifest_structure(manifest)

    arrays: Dict[str, np.ndarray] = {}
    file_identities = {}
    for name in ARRAY_FILENAMES:
        entry = manifest["files"][name]
        array_path = (root / entry["path"]).resolve()
        try:
            array_path.relative_to(root)
        except ValueError:
            raise ArtifactValidationError("artifact path escapes bundle: {}".format(array_path))
        if not array_path.is_file():
            raise ArtifactValidationError("artifact file is missing: {}".format(array_path))
        stat = array_path.stat()
        file_identities[name] = (array_path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if verify_checksums and validation_mode != "fast" and _sha256(array_path) != entry["sha256"]:
            raise ArtifactValidationError("checksum mismatch for {}".format(name))
        arrays[name] = np.load(array_path, mmap_mode=mmap_mode, allow_pickle=False)

    cache_key = (str(root), hashlib.sha256(manifest_bytes).hexdigest(),
                 tuple(file_identities.items()), validation_chunk_rows)
    cached = validation_mode == "fast" and verify_checksums and cache_key in _FAST_VALIDATION_CACHE
    if cached:
        print("Fast validation: reuse complete in-process validation; file identities unchanged", flush=True)
    else:
        if validation_mode == "fast" and verify_checksums:
            def check_file(name):
                print("Fast validation hashing " + name, flush=True)
                if _sha256(file_identities[name][0]) != manifest["files"][name]["sha256"]:
                    raise ArtifactValidationError("checksum mismatch for " + name)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(check_file, ARRAY_FILENAMES))
        if validation_mode in ("bounded", "fast"):
            from .bounded_artifacts import validate_artifact_bundle as validate_bounded
            validate_bounded(manifest, arrays, validation_chunk_rows, validation_scratch_dir,
                             fast=validation_mode == "fast")
        else:
            validate_artifact_bundle(manifest, arrays)
    for array_path, dev, ino, size, mtime, ctime in file_identities.values():
        stat = array_path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (dev, ino, size, mtime, ctime):
            raise ArtifactValidationError("artifact changed during validation")
    if manifest_path.read_bytes() != manifest_bytes:
        raise ArtifactValidationError("manifest changed during validation")
    if validation_mode == "fast" and verify_checksums:
        if len(_FAST_VALIDATION_CACHE) >= 8:
            _FAST_VALIDATION_CACHE.clear()
        _FAST_VALIDATION_CACHE.add(cache_key)
    return ArtifactBundle(root=root, manifest=manifest, arrays=arrays)
