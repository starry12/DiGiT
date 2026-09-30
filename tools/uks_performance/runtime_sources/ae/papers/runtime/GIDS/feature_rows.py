"""Resolve logical input nodes to physical feature rows for GIDS reads."""

from typing import Optional, Tuple

import torch


DIGIT_STORAGE_ROW = "digit_storage_row"
DIGIT_STORAGE_IS_GROUP = "digit_storage_is_group"
DGL_NID = "_ID"
FEATURE_INDEX_MODES = ("auto", "logical", "explicit", "legacy_map")


def _explicit_rows_from_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        return None, None
    blocks = batch[2]
    if not blocks:
        return None, None
    outermost = blocks[0]
    srcdata = getattr(outermost, "srcdata", None)
    if srcdata is None or DIGIT_STORAGE_ROW not in srcdata:
        return None, None
    block_logical_nodes = srcdata[DGL_NID] if DGL_NID in srcdata else None
    return srcdata[DIGIT_STORAGE_ROW], block_logical_nodes


def _explicit_group_flags_from_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        return None
    blocks = batch[2]
    if not blocks:
        return None
    srcdata = getattr(blocks[0], "srcdata", None)
    if srcdata is None or DIGIT_STORAGE_IS_GROUP not in srcdata:
        return None
    return srcdata[DIGIT_STORAGE_IS_GROUP]


def resolve_feature_rows(
    batch,
    *,
    mode: str = "auto",
    graph_reorganize: bool = False,
    feature_map: Optional[torch.Tensor] = None,
    target_device=None,
    strict: bool = False,
) -> Tuple[torch.Tensor, str]:
    """Return contiguous int64 physical rows and the selected index source.

    ``batch[0]`` and DGL block NIDs remain logical IDs. Only the returned tensor
    is passed to BaM as a physical storage-row array.
    """

    if mode not in FEATURE_INDEX_MODES:
        raise ValueError("feature index mode must be one of {}".format(FEATURE_INDEX_MODES))
    if not isinstance(batch, (tuple, list)) or not batch:
        raise TypeError("homogeneous GIDS batch must be a non-empty tuple or list")
    logical_nodes = batch[0]
    if not torch.is_tensor(logical_nodes) or logical_nodes.ndim != 1:
        raise TypeError("batch[0] must be a one-dimensional logical-node tensor")

    explicit_rows, block_logical_nodes = _explicit_rows_from_batch(batch)
    use_explicit = mode == "explicit" or (mode == "auto" and explicit_rows is not None)
    use_legacy = mode == "legacy_map" or (
        mode == "auto" and explicit_rows is None and graph_reorganize
    )

    if use_explicit:
        if explicit_rows is None:
            raise RuntimeError(
                "explicit feature rows were requested but digit_storage_row is absent"
            )
        if not torch.is_tensor(explicit_rows) or explicit_rows.ndim != 1:
            raise TypeError("digit_storage_row must be a one-dimensional tensor")
        if explicit_rows.numel() != logical_nodes.numel():
            raise ValueError("digit_storage_row is not aligned with batch input nodes")
        if block_logical_nodes is not None:
            if block_logical_nodes.shape != logical_nodes.shape:
                raise ValueError("outermost block NIDs are not aligned with batch input nodes")
            if strict and not torch.equal(
                block_logical_nodes.to(logical_nodes.device), logical_nodes
            ):
                raise ValueError("outermost block NIDs differ from batch input nodes")
        rows = explicit_rows
        source = "explicit"
    elif use_legacy:
        if feature_map is None:
            raise RuntimeError("legacy_map mode requires feature_map")
        if not torch.is_tensor(feature_map) or feature_map.ndim != 1:
            raise TypeError("feature_map must be a one-dimensional tensor")
        map_nodes = logical_nodes.to(feature_map.device)
        rows = feature_map[map_nodes]
        source = "legacy_map"
    else:
        rows = logical_nodes
        source = "logical"

    if not torch.is_tensor(rows) or rows.ndim != 1:
        raise TypeError("resolved feature rows must be a one-dimensional tensor")
    rows = rows.to(device=target_device, dtype=torch.int64).contiguous()
    if strict and rows.numel() and bool(torch.any(rows < 0).item()):
        raise ValueError("resolved feature rows cannot be negative")
    return rows, source


def resolve_feature_accesses(
    batch,
    *,
    mode: str = "auto",
    graph_reorganize: bool = False,
    feature_map: Optional[torch.Tensor] = None,
    target_device=None,
    strict: bool = False,
    require_group_flags: bool = False,
    allow_implicit_raw_flags: bool = False,
):
    """Return physical rows, aligned group/raw tags, and index source."""

    rows, source = resolve_feature_rows(
        batch,
        mode=mode,
        graph_reorganize=graph_reorganize,
        feature_map=feature_map,
        target_device=target_device,
        strict=strict,
    )
    group_flags = _explicit_group_flags_from_batch(batch)
    if source == "explicit" and group_flags is not None:
        if not torch.is_tensor(group_flags) or group_flags.ndim != 1:
            raise TypeError("digit_storage_is_group must be a one-dimensional tensor")
        if group_flags.numel() != rows.numel():
            raise ValueError("digit_storage_is_group is not aligned with feature rows")
        flags = group_flags.to(device=target_device, dtype=torch.bool).contiguous()
    else:
        if require_group_flags and not (
            allow_implicit_raw_flags and source == "legacy_map"
        ):
            raise RuntimeError(
                "mixed I/O requires digit_storage_is_group on every explicit batch"
            )
        flags = torch.zeros(rows.shape, dtype=torch.bool, device=target_device)
    return rows, flags, source
