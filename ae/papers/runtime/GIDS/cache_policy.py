"""Resolve the GPU cache-entry geometry used by mixed DiGiT I/O."""

from collections.abc import Mapping


MIXED_CACHE_ENTRY_POLICIES = ("auto", "split", "coupled")


def _positive_geometry_value(geometry, name):
    try:
        value = int(geometry[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "mixed-I/O geometry is missing a valid {}".format(name)
        ) from error
    if value <= 0:
        raise ValueError("mixed-I/O geometry {} must be positive".format(name))
    return value


def resolve_mixed_cache_entry_policy(
    requested_policy="auto",
    *,
    mixed_io=False,
    mixed_io_geometry=None,
    page_size=4096,
    num_ssd=1,
    legacy_split_entries=None,
):
    """Return a fail-closed requested/resolved cache-entry policy contract.

    ``legacy_split_entries`` preserves the former boolean API.  ``True`` is an
    explicit request for split entries and ``False`` is an explicit request
    for coupled entries.  It may be combined with ``auto`` but must agree with
    any explicitly named non-auto policy.

    The current native split path atomically acquires two independently tagged
    4-KiB entries and may merge their cold fill into one 8-KiB command.  Auto
    selection therefore accepts exactly that validated geometry.  A one-subrow
    slot is already one physical cache entry.  Wider split spans fail closed
    until the native acquisition primitive is generalized and validated.
    """

    requested_policy = str(requested_policy)
    if requested_policy not in MIXED_CACHE_ENTRY_POLICIES:
        raise ValueError(
            "mixed_cache_entry_policy must be one of {}".format(
                MIXED_CACHE_ENTRY_POLICIES
            )
        )
    if legacy_split_entries is not None and not isinstance(
        legacy_split_entries, bool
    ):
        raise TypeError("legacy mixed_cache_split_entries must be bool or None")

    effective_policy = requested_policy
    resolution_source = "explicit" if requested_policy != "auto" else "geometry"
    if legacy_split_entries is not None:
        alias_policy = "split" if legacy_split_entries else "coupled"
        if requested_policy not in ("auto", alias_policy):
            raise ValueError(
                "mixed_cache_split_entries contradicts "
                "mixed_cache_entry_policy={}".format(requested_policy)
            )
        effective_policy = alias_policy
        resolution_source = "legacy_boolean_alias"

    page_size = int(page_size)
    num_ssd = int(num_ssd)
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if num_ssd <= 0:
        raise ValueError("num_ssd must be positive")

    if not mixed_io:
        if effective_policy == "split":
            raise ValueError("split cache entries require mixed_io=True")
        return {
            "requested": requested_policy,
            "effective_requested": effective_policy,
            "resolved": "disabled",
            "resolution_source": resolution_source,
            "cache_entry_bytes": page_size,
            "entries_per_storage_slot": 1,
            "group_merge_width": 1,
        }

    if mixed_io_geometry is None:
        if effective_policy == "split":
            raise ValueError(
                "split cache entries require versioned mixed-I/O geometry"
            )
        # Backward-compatible Phase 8D geometry.  It predates independently
        # tagged entries and is coupled unless a versioned geometry says
        # otherwise.
        return {
            "requested": requested_policy,
            "effective_requested": effective_policy,
            "resolved": "coupled",
            "resolution_source": (
                "legacy_geometry" if effective_policy == "auto"
                else resolution_source
            ),
            "cache_entry_bytes": page_size,
            "entries_per_storage_slot": 1,
            "group_merge_width": 1,
        }

    if not isinstance(mixed_io_geometry, Mapping):
        raise TypeError("mixed_io_geometry must be a mapping")
    slot_bytes = _positive_geometry_value(
        mixed_io_geometry, "cache_slot_bytes"
    )
    transfer_bytes = _positive_geometry_value(
        mixed_io_geometry, "minimum_transfer_bytes"
    )
    subrow_count = _positive_geometry_value(mixed_io_geometry, "subrow_count")
    if slot_bytes != page_size:
        raise ValueError("mixed-I/O cache slot does not match page_size")
    if slot_bytes % transfer_bytes:
        raise ValueError("mixed-I/O cache slot is not transfer aligned")
    if subrow_count != slot_bytes // transfer_bytes:
        raise ValueError("mixed-I/O subrow count is inconsistent")

    if effective_policy == "auto":
        if subrow_count == 1:
            resolved = "coupled"
        elif (
            subrow_count == 2
            and transfer_bytes == 4096
            and slot_bytes == 8192
            and num_ssd == 1
        ):
            resolved = "split"
        else:
            raise ValueError(
                "auto cache-entry policy has no validated implementation for "
                "{} subrows of {} bytes in a {}-byte slot; request coupled "
                "explicitly or add a validated split-span implementation"
                .format(subrow_count, transfer_bytes, slot_bytes)
            )
    else:
        resolved = effective_policy

    if resolved == "split" and not (
        subrow_count == 2
        and transfer_bytes == 4096
        and slot_bytes == 8192
        and num_ssd == 1
    ):
        raise ValueError(
            "split cache currently requires one SSD and two 4-KiB entries "
            "per 8-KiB storage slot"
        )

    entry_bytes = transfer_bytes if resolved == "split" else slot_bytes
    entries_per_slot = slot_bytes // entry_bytes
    return {
        "requested": requested_policy,
        "effective_requested": effective_policy,
        "resolved": resolved,
        "resolution_source": resolution_source,
        "cache_entry_bytes": entry_bytes,
        "entries_per_storage_slot": entries_per_slot,
        "group_merge_width": entries_per_slot if resolved == "split" else 1,
    }
