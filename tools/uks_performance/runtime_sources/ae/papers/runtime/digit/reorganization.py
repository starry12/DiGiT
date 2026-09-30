"""Deterministic implementation of DiGiT graph and feature reorganization.

The graph is represented as CSC: column ``v`` contains the logical source
nodes with edges into destination/owner ``v``. Grouping therefore replaces
``member -> owner`` edges with one ``supernode -> owner`` edge.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .artifacts import (
    ARRAY_FILENAMES,
    ArtifactBundle,
    ArtifactValidationError,
    MANIFEST_FILENAME,
    finalize_artifact_bundle,
)


@dataclass(frozen=True)
class GroupingResult:
    """Groups and the exact original CSC entries replaced by them."""

    group_members: np.ndarray
    group_owner: np.ndarray
    num_primary_groups: int
    owner_order: np.ndarray
    covered_edge_mask: np.ndarray
    primary_grouped_nodes: np.ndarray
    statistics: Mapping[str, Any]

    @property
    def num_groups(self) -> int:
        return int(self.group_members.shape[0])

    @property
    def num_replica_groups(self) -> int:
        return self.num_groups - self.num_primary_groups


@dataclass(frozen=True)
class StorageLayout:
    """Mappings between group/logical IDs and feature storage rows."""

    group_storage_base: np.ndarray
    storage_to_node: np.ndarray
    node_to_primary_row: np.ndarray


def _validate_original_csc(
    indptr: np.ndarray,
    indices: np.ndarray,
    num_nodes: int,
    *,
    require_unique_neighbors: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    if not np.issubdtype(indptr.dtype, np.integer):
        raise ArtifactValidationError("CSC indptr must use an integer dtype")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ArtifactValidationError("CSC indices must use an integer dtype")
    if num_nodes <= 0 or indptr.shape != (num_nodes + 1,):
        raise ArtifactValidationError("original CSC indptr must have num_nodes + 1 entries")
    if indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
        raise ArtifactValidationError("original CSC indptr must start at zero and be monotonic")
    if int(indptr[-1]) != indices.size:
        raise ArtifactValidationError("original CSC indptr[-1] must equal indices size")
    if indices.size and (indices.min() < 0 or indices.max() >= num_nodes):
        raise ArtifactValidationError("original CSC contains an invalid logical node ID")
    if require_unique_neighbors:
        for owner in range(num_nodes):
            neighbors = indices[int(indptr[owner]) : int(indptr[owner + 1])]
            if neighbors.size != np.unique(neighbors).size:
                raise ArtifactValidationError(
                    "parallel edges are not supported in v1 grouping (owner {})".format(owner)
                )
    return indptr, indices


def _one_position_per_logical_node(
    positions: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Keep one edge occurrence per node; other parallel edges remain raw."""

    if not positions.size:
        return positions
    _, first_occurrences = np.unique(indices[positions], return_index=True)
    return positions[first_occurrences]


def edge_index_to_csc(edge_index: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a ``[2, E]`` or ``[E, 2]`` logical ``(src, dst)`` array to CSC."""

    edges = np.asarray(edge_index)
    if edges.ndim != 2:
        raise ArtifactValidationError("edge_index must be a two-dimensional array")
    if edges.shape[0] == 2:
        sources, destinations = edges[0], edges[1]
    elif edges.shape[1] == 2:
        sources, destinations = edges[:, 0], edges[:, 1]
    else:
        raise ArtifactValidationError("edge_index must have shape [2, E] or [E, 2]")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ArtifactValidationError("edge_index must use an integer dtype")
    if sources.size and (
        sources.min() < 0
        or sources.max() >= num_nodes
        or destinations.min() < 0
        or destinations.max() >= num_nodes
    ):
        raise ArtifactValidationError("edge_index contains an invalid logical node ID")

    order = np.argsort(destinations, kind="stable")
    sorted_sources = np.asarray(sources[order], dtype=np.int64)
    counts = np.bincount(np.asarray(destinations, dtype=np.int64), minlength=num_nodes)
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    _validate_original_csc(indptr, sorted_sources, num_nodes)
    return indptr, sorted_sources


def _hot_mask(hot_nodes: Optional[np.ndarray], num_nodes: int) -> np.ndarray:
    mask = np.zeros(num_nodes, dtype=bool)
    if hot_nodes is None:
        return mask
    values = np.asarray(hot_nodes)
    if values.dtype == np.bool_:
        if values.shape != (num_nodes,):
            raise ArtifactValidationError("boolean hot_nodes must have shape [num_nodes]")
        return values.copy()
    if not np.issubdtype(values.dtype, np.integer):
        raise ArtifactValidationError("hot_nodes must contain logical integer IDs")
    values = values.reshape(-1).astype(np.int64, copy=False)
    if values.size and (values.min() < 0 or values.max() >= num_nodes):
        raise ArtifactValidationError("hot_nodes contains an invalid logical node ID")
    mask[values] = True
    return mask


def build_groups(
    indptr: np.ndarray,
    indices: np.ndarray,
    num_nodes: int,
    *,
    group_size: int = 2,
    replication_ratio: float = 0.2,
    hot_nodes: Optional[np.ndarray] = None,
    seed: int = 0,
) -> GroupingResult:
    """Build disjoint primary groups followed by budgeted replica groups.

    Primary members are globally unique. Replica groups are formed only from
    still-uncovered adjacency entries whose logical nodes already belong to a
    primary group, so every replica row is an actual additional SSD copy.
    """

    indptr, indices = _validate_original_csc(indptr, indices, num_nodes)
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)):
        raise ArtifactValidationError("group_size must be an integer")
    group_size = int(group_size)
    if group_size <= 0:
        raise ArtifactValidationError("group_size must be positive")
    if not np.isfinite(replication_ratio) or replication_ratio < 0:
        raise ArtifactValidationError("replication_ratio must be finite and non-negative")

    hot = _hot_mask(hot_nodes, num_nodes)
    eligible_edges = ~hot[indices]
    prefix = np.empty(indices.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(eligible_edges, dtype=np.int64, out=prefix[1:])
    eligible_degree = prefix[indptr[1:].astype(np.int64)] - prefix[indptr[:-1].astype(np.int64)]
    node_ids = np.arange(num_nodes, dtype=np.int64)
    owner_order = np.lexsort((node_ids, -eligible_degree)).astype(np.int64, copy=False)

    rng = np.random.default_rng(seed)
    primary_used = np.zeros(num_nodes, dtype=bool)
    covered = np.zeros(indices.size, dtype=bool)
    replica_row_budget = int(np.floor(float(replication_ratio) * num_nodes))
    max_primary_groups = num_nodes // group_size
    max_replica_groups = replica_row_budget // group_size
    group_capacity = max_primary_groups + max_replica_groups
    group_members = np.empty((group_capacity, group_size), dtype=np.int64)
    group_owner = np.empty(group_capacity, dtype=np.int64)
    num_primary = 0

    for owner in owner_order:
        start, end = int(indptr[owner]), int(indptr[owner + 1])
        positions = np.arange(start, end, dtype=np.int64)
        if positions.size < group_size:
            continue
        neighbors = indices[start:end]
        available = (~hot[neighbors]) & (~primary_used[neighbors])
        candidates = _one_position_per_logical_node(positions[available], indices)
        take = (candidates.size // group_size) * group_size
        if not take:
            continue
        rng.shuffle(candidates)
        candidates = candidates[:take]
        for offset in range(0, take, group_size):
            group_positions = candidates[offset : offset + group_size]
            members = np.asarray(indices[group_positions], dtype=np.int64)
            group_members[num_primary] = members
            group_owner[num_primary] = owner
            num_primary += 1
            primary_used[members] = True
            covered[group_positions] = True

    num_replica = 0

    for owner in owner_order:
        if num_replica >= max_replica_groups:
            break
        start, end = int(indptr[owner]), int(indptr[owner + 1])
        positions = np.arange(start, end, dtype=np.int64)
        if positions.size < group_size:
            continue
        neighbors = indices[start:end]
        available = (~hot[neighbors]) & primary_used[neighbors] & (~covered[positions])
        candidates = _one_position_per_logical_node(positions[available], indices)
        remaining_groups = max_replica_groups - num_replica
        take_groups = min(candidates.size // group_size, remaining_groups)
        if not take_groups:
            continue
        rng.shuffle(candidates)
        candidates = candidates[: take_groups * group_size]
        for offset in range(0, candidates.size, group_size):
            group_positions = candidates[offset : offset + group_size]
            group_id = num_primary + num_replica
            group_members[group_id] = indices[group_positions]
            group_owner[group_id] = owner
            num_replica += 1
            covered[group_positions] = True

    num_groups = num_primary + num_replica
    group_members = group_members[:num_groups].copy()
    group_owner = group_owner[:num_groups].copy()
    covered_primary_edges = num_primary * group_size
    covered_replica_edges = num_replica * group_size
    statistics: Dict[str, Any] = {
        "num_hot_nodes": int(hot.sum()),
        "eligible_edges": int(eligible_edges.sum()),
        "num_primary_groups": num_primary,
        "num_replica_groups": num_replica,
        "primary_grouped_nodes": int(primary_used.sum()),
        "replica_rows_used": covered_replica_edges,
        "replica_row_budget": replica_row_budget,
        "covered_edges": covered_primary_edges + covered_replica_edges,
        "group_coverage": (
            float(covered_primary_edges + covered_replica_edges) / float(indices.size)
            if indices.size
            else 0.0
        ),
    }
    return GroupingResult(
        group_members=group_members,
        group_owner=group_owner,
        num_primary_groups=num_primary,
        owner_order=owner_order,
        covered_edge_mask=covered,
        primary_grouped_nodes=primary_used,
        statistics=statistics,
    )


def rewrite_csc_with_supernodes(
    indptr: np.ndarray,
    indices: np.ndarray,
    num_nodes: int,
    grouping: GroupingResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace every grouped edge set with its graph-side supernode."""

    indptr, indices = _validate_original_csc(indptr, indices, num_nodes)
    if grouping.covered_edge_mask.shape != indices.shape:
        raise ArtifactValidationError("covered_edge_mask shape does not match CSC indices")
    group_counts = np.bincount(grouping.group_owner, minlength=num_nodes)
    group_order = np.argsort(grouping.group_owner, kind="stable")
    group_offsets = np.empty(num_nodes + 1, dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(group_counts, out=group_offsets[1:])

    counts = np.empty(num_nodes, dtype=np.int64)
    for owner in range(num_nodes):
        start, end = int(indptr[owner]), int(indptr[owner + 1])
        raw_count = int((~grouping.covered_edge_mask[start:end]).sum())
        counts[owner] = raw_count + group_counts[owner]

    num_groups = grouping.num_groups
    rewritten_indptr = np.empty(num_nodes + num_groups + 1, dtype=np.int64)
    rewritten_indptr[0] = 0
    np.cumsum(counts, out=rewritten_indptr[1 : num_nodes + 1])
    rewritten_indptr[num_nodes + 1 :] = rewritten_indptr[num_nodes]
    rewritten_indices = np.empty(int(rewritten_indptr[-1]), dtype=np.int64)

    for owner in range(num_nodes):
        original_start, original_end = int(indptr[owner]), int(indptr[owner + 1])
        output_start = int(rewritten_indptr[owner])
        raw = indices[original_start:original_end][
            ~grouping.covered_edge_mask[original_start:original_end]
        ]
        rewritten_indices[output_start : output_start + raw.size] = raw
        group_start, group_end = int(group_offsets[owner]), int(group_offsets[owner + 1])
        if group_end > group_start:
            group_ids = group_order[group_start:group_end]
            rewritten_indices[
                output_start + raw.size : output_start + raw.size + group_ids.size
            ] = num_nodes + group_ids
    return rewritten_indptr, rewritten_indices


def expand_reorganized_csc(
    indptr: np.ndarray,
    indices: np.ndarray,
    num_nodes: int,
    group_members: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand supernodes for correctness testing and offline verification."""

    num_groups = int(group_members.shape[0])
    if indptr.shape != (num_nodes + num_groups + 1,):
        raise ArtifactValidationError("reorganized CSC indptr shape is inconsistent")
    columns = []
    counts = np.empty(num_nodes, dtype=np.int64)
    for owner in range(num_nodes):
        expanded = []
        for graph_id in indices[int(indptr[owner]) : int(indptr[owner + 1])]:
            graph_id = int(graph_id)
            if graph_id < num_nodes:
                expanded.append(graph_id)
            else:
                group_id = graph_id - num_nodes
                if not 0 <= group_id < num_groups:
                    raise ArtifactValidationError("invalid supernode while expanding CSC")
                expanded.extend(np.asarray(group_members[group_id], dtype=np.int64).tolist())
        column = np.asarray(expanded, dtype=np.int64)
        columns.append(column)
        counts[owner] = column.size
    expanded_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    expanded_indptr[0] = 0
    np.cumsum(counts, out=expanded_indptr[1:])
    expanded_indices = (
        np.concatenate(columns) if int(expanded_indptr[-1]) else np.empty(0, dtype=np.int64)
    )
    return expanded_indptr, expanded_indices


def assert_csc_adjacency_equivalent(
    original_indptr: np.ndarray,
    original_indices: np.ndarray,
    expanded_indptr: np.ndarray,
    expanded_indices: np.ndarray,
) -> None:
    """Assert equality of every CSC neighbor multiset, ignoring list order."""

    if not np.array_equal(original_indptr, expanded_indptr):
        raise ArtifactValidationError("expanded CSC degree sequence differs from original")
    for owner in range(original_indptr.size - 1):
        start, end = int(original_indptr[owner]), int(original_indptr[owner + 1])
        if not np.array_equal(
            np.sort(original_indices[start:end]), np.sort(expanded_indices[start:end])
        ):
            raise ArtifactValidationError(
                "expanded CSC neighbors differ from original for owner {}".format(owner)
            )


def assert_rewritten_csc_equivalent(
    original_indptr: np.ndarray,
    original_indices: np.ndarray,
    rewritten_indptr: np.ndarray,
    rewritten_indices: np.ndarray,
    num_nodes: int,
    group_members: np.ndarray,
) -> None:
    """Verify rewritten adjacency column by column with bounded extra memory."""

    num_groups = int(group_members.shape[0])
    if rewritten_indptr.shape != (num_nodes + num_groups + 1,):
        raise ArtifactValidationError("reorganized CSC indptr shape is inconsistent")
    for owner in range(num_nodes):
        original = original_indices[
            int(original_indptr[owner]) : int(original_indptr[owner + 1])
        ]
        rewritten = rewritten_indices[
            int(rewritten_indptr[owner]) : int(rewritten_indptr[owner + 1])
        ]
        raw = rewritten[rewritten < num_nodes]
        supernodes = rewritten[rewritten >= num_nodes] - num_nodes
        if supernodes.size and (supernodes.min() < 0 or supernodes.max() >= num_groups):
            raise ArtifactValidationError("reorganized CSC contains an invalid supernode")
        if supernodes.size:
            expanded = np.concatenate([raw, group_members[supernodes].reshape(-1)])
        else:
            expanded = raw
        if not np.array_equal(np.sort(original), np.sort(expanded)):
            raise ArtifactValidationError(
                "rewritten CSC does not recover original owner {}".format(owner)
            )


def build_storage_layout(
    grouping: GroupingResult,
    num_nodes: int,
    *,
    alignment_rows: int,
    hot_nodes: Optional[np.ndarray] = None,
) -> StorageLayout:
    """Lay out groups, page-packed hot raw nodes, then other raw nodes."""

    if alignment_rows <= 0:
        raise ArtifactValidationError("alignment_rows must be positive")
    group_size = int(grouping.group_members.shape[1])
    if group_size > alignment_rows:
        raise ArtifactValidationError(
            "group_size cannot exceed one aligned cache slot"
        )
    bases = np.empty(grouping.num_groups, dtype=np.int64)
    cursor = 0
    for group_id in range(grouping.num_groups):
        cursor = ((cursor + alignment_rows - 1) // alignment_rows) * alignment_rows
        bases[group_id] = cursor
        # Reserve the complete rounded cache slot. Padding rows stay -1 and
        # can never be reused by a raw feature or another group.
        cursor += alignment_rows

    hot = _hot_mask(hot_nodes, num_nodes)
    if np.any(grouping.primary_grouped_nodes & hot):
        raise ArtifactValidationError("hot nodes cannot be primary group members")
    remaining = ~grouping.primary_grouped_nodes
    hot_remaining = np.flatnonzero(remaining & hot).astype(np.int64)
    cold_remaining = np.flatnonzero(remaining & ~hot).astype(np.int64)
    hot_start = cursor
    hot_end = hot_start + hot_remaining.size
    cold_start = ((hot_end + alignment_rows - 1) // alignment_rows) * alignment_rows
    raw_end = cold_start + cold_remaining.size
    num_storage_rows = ((raw_end + alignment_rows - 1) // alignment_rows) * alignment_rows
    storage_to_node = np.full(num_storage_rows, -1, dtype=np.int64)
    node_to_primary = np.full(num_nodes, -1, dtype=np.int64)

    row_offsets = np.arange(group_size, dtype=np.int64)
    for group_id in range(grouping.num_groups):
        rows = bases[group_id] + row_offsets
        members = grouping.group_members[group_id]
        storage_to_node[rows] = members
        if group_id < grouping.num_primary_groups:
            node_to_primary[members] = rows
    hot_rows = hot_start + np.arange(hot_remaining.size, dtype=np.int64)
    storage_to_node[hot_rows] = hot_remaining
    node_to_primary[hot_remaining] = hot_rows
    cold_rows = cold_start + np.arange(cold_remaining.size, dtype=np.int64)
    storage_to_node[cold_rows] = cold_remaining
    node_to_primary[cold_remaining] = cold_rows

    if np.any(node_to_primary < 0):
        raise ArtifactValidationError("storage layout did not assign every logical node")
    return StorageLayout(
        group_storage_base=bases,
        storage_to_node=storage_to_node,
        node_to_primary_row=node_to_primary,
    )


def stream_reordered_features(
    source_features: np.ndarray,
    storage_to_node: np.ndarray,
    output_path: os.PathLike,
    *,
    batch_rows: int = 16384,
) -> np.memmap:
    """Write reordered features in bounded batches without materializing them."""

    source = np.asarray(source_features)
    if source.ndim != 2 or source.dtype.hasobject:
        raise ArtifactValidationError("source_features must be a numeric [N, D] array")
    if batch_rows <= 0:
        raise ArtifactValidationError("batch_rows must be positive")
    mapping = np.asarray(storage_to_node)
    if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
        raise ArtifactValidationError("storage_to_node must be a one-dimensional integer array")
    if mapping.size and (mapping.min() < -1 or mapping.max() >= source.shape[0]):
        raise ArtifactValidationError("storage_to_node is invalid for source_features")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError("reordered feature file already exists: {}".format(path))
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=source.dtype,
        shape=(mapping.size, source.shape[1]),
    )
    for start in range(0, mapping.size, batch_rows):
        end = min(start + batch_rows, mapping.size)
        node_ids = mapping[start:end]
        destination = output[start:end]
        destination[...] = 0
        valid = node_ids >= 0
        if np.any(valid):
            destination[valid] = source[node_ids[valid]]
    output.flush()
    return output


def reorganize_to_bundle(
    indptr: np.ndarray,
    indices: np.ndarray,
    source_features: np.ndarray,
    output_dir: os.PathLike,
    *,
    dataset_name: str,
    dataset_size: str,
    group_size: int = 2,
    replication_ratio: float = 0.2,
    page_size: int = 8192,
    minimum_transfer_bytes: Optional[int] = None,
    target_request_bytes: Optional[int] = None,
    hot_nodes: Optional[np.ndarray] = None,
    seed: int = 0,
    feature_batch_rows: int = 16384,
    source: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ArtifactBundle:
    """Run graph reorganization and publish one validated v2 artifact bundle."""

    features = np.asarray(source_features)
    if features.ndim != 2:
        raise ArtifactValidationError("source_features must have shape [N, D]")
    num_nodes, feature_dim = features.shape
    indptr, indices = _validate_original_csc(indptr, indices, num_nodes)
    row_bytes = int(features.dtype.itemsize * feature_dim)
    if page_size <= 0 or page_size % row_bytes:
        raise ArtifactValidationError("page_size must be a positive multiple of feature row bytes")
    alignment_rows = page_size // row_bytes

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    conflicts = [MANIFEST_FILENAME] + list(ARRAY_FILENAMES.values())
    existing = [name for name in conflicts if (output / name).exists()]
    if existing:
        raise FileExistsError("artifact output already contains: {}".format(", ".join(existing)))

    grouping = build_groups(
        indptr,
        indices,
        num_nodes,
        group_size=group_size,
        replication_ratio=replication_ratio,
        hot_nodes=hot_nodes,
        seed=seed,
    )
    rewritten_indptr, rewritten_indices = rewrite_csc_with_supernodes(
        indptr, indices, num_nodes, grouping
    )
    assert_rewritten_csc_equivalent(
        indptr,
        indices,
        rewritten_indptr,
        rewritten_indices,
        num_nodes,
        grouping.group_members,
    )
    layout = build_storage_layout(
        grouping,
        num_nodes,
        alignment_rows=alignment_rows,
        hot_nodes=hot_nodes,
    )

    arrays = {
        "group_members": grouping.group_members,
        "group_owner": grouping.group_owner,
        "group_storage_base": layout.group_storage_base,
        "storage_to_node": layout.storage_to_node,
        "node_to_primary_row": layout.node_to_primary_row,
        "supernode_to_group": np.arange(grouping.num_groups, dtype=np.int64),
        "reordered_indptr": rewritten_indptr,
        "reordered_indices": rewritten_indices,
    }
    for name, array in arrays.items():
        np.save(output / ARRAY_FILENAMES[name], array, allow_pickle=False)
    stream_reordered_features(
        features,
        layout.storage_to_node,
        output / ARRAY_FILENAMES["reordered_features"],
        batch_rows=feature_batch_rows,
    )

    hot = _hot_mask(hot_nodes, num_nodes)
    hot_rows = np.sort(layout.node_to_primary_row[hot])
    hot_layout = {
        "count": int(hot_rows.size),
        "first_storage_row": int(hot_rows[0]) if hot_rows.size else None,
        "last_storage_row": int(hot_rows[-1]) if hot_rows.size else None,
        "page_aligned": bool(
            not hot_rows.size
            or (
                hot_rows[0] % alignment_rows == 0
                and hot_rows.size % alignment_rows == 0
                and np.array_equal(
                    hot_rows,
                    np.arange(hot_rows[0], hot_rows[0] + hot_rows.size, dtype=np.int64),
                )
            )
        ),
    }
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "phase": 2,
            "seed": int(seed),
            "grouping_statistics": dict(grouping.statistics),
            "rewritten_edges": int(rewritten_indices.size),
            "storage_expansion_ratio": float(layout.storage_to_node.size) / float(num_nodes),
            "hot_storage_layout": hot_layout,
        }
    )
    return finalize_artifact_bundle(
        output,
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        num_nodes=num_nodes,
        num_edges=int(indices.size),
        feature_dim=feature_dim,
        group_size=group_size,
        replication_ratio=replication_ratio,
        num_primary_groups=grouping.num_primary_groups,
        page_size=page_size,
        source=source,
        metadata=artifact_metadata,
        minimum_transfer_bytes=minimum_transfer_bytes,
        target_request_bytes=target_request_bytes,
    )
