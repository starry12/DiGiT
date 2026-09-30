import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import dgl
import torch


def _git_value(repo_root, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def collect_run_metadata(args, graph, repo_root):
    repo_root = Path(repo_root).resolve()
    git_status = _git_value(repo_root, "status", "--porcelain")
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "logical_device": torch.cuda.current_device(),
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        }

    config = {
        key: _json_safe(value)
        for key, value in vars(args).items()
        if key != "feat_map"
    }
    return {
        "schema_version": 1,
        "system": "GIDS",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "config": config,
        "dataset": {
            "name": args.data,
            "size": args.dataset_size,
            "num_nodes": graph.num_nodes(),
            "num_edges": graph.num_edges(),
            "feature_dim": args.emb_size,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "dgl": dgl.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": gpu,
        },
        "source": {
            "repo_root": str(repo_root),
            "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
        },
    }


def io_stats(tensor):
    values = tensor.reshape(-1).tolist()
    accesses, misses, hits = (int(values[0]), int(values[1]), int(values[2]))
    return {
        "accesses": accesses,
        "misses": misses,
        "hits": hits,
        "miss_rate": misses / accesses if accesses else 0.0,
        "hit_rate": hits / accesses if accesses else 0.0,
    }


def gpu_cache_interval(start, end):
    """Return a reconciled interval between cumulative GPU-cache snapshots."""

    if start["policy"] != end["policy"]:
        raise ValueError("GPU-cache policy changed within an interval")
    if start["capacity_pages"] != end["capacity_pages"]:
        raise ValueError("GPU-cache capacity changed within an interval")
    counters = {}
    for key in (
        "requests", "hits", "inserts", "partial_fills", "evictions",
        "fifo_ticket",
    ):
        delta = int(end.get(key, 0)) - int(start.get(key, 0))
        if delta < 0:
            raise ValueError("GPU-cache counter decreased: {}".format(key))
        counters[key] = delta
    physical_insert_start = int(start.get("physical_inserts", start["inserts"]))
    physical_insert_end = int(end.get("physical_inserts", end["inserts"]))
    physical_inserts = physical_insert_end - physical_insert_start
    if physical_inserts < 0:
        raise ValueError("GPU-cache physical insert counter decreased")
    counters["physical_inserts"] = physical_inserts
    resident_start = int(start["resident_pages"])
    resident_end = int(end["resident_pages"])
    resident_delta = resident_end - resident_start
    if resident_delta < 0:
        raise ValueError("GPU-cache residency decreased unexpectedly")
    if counters["requests"] != (
        counters["hits"] + counters["inserts"] + counters["partial_fills"]
    ):
        raise ValueError("GPU-cache interval requests do not reconcile")
    if counters["physical_inserts"] != counters["evictions"] + resident_delta:
        raise ValueError("GPU-cache interval physical inserts do not reconcile")
    if start["policy"] == "fifo" and counters["fifo_ticket"] != counters["physical_inserts"]:
        raise ValueError("FIFO interval ticket does not reconcile")
    capacity = int(end["capacity_pages"])
    return {
        "policy": end["policy"],
        "capacity_pages": capacity,
        **counters,
        "hit_rate": (
            counters["hits"] / counters["requests"]
            if counters["requests"] else 0.0
        ),
        "resident_pages_start": resident_start,
        "resident_pages_end": resident_end,
        "resident_pages_delta": resident_delta,
        "occupancy_rate_end": resident_end / capacity if capacity else 0.0,
        "reconciled": True,
    }


def cpu_staging_interval(start, end):
    """Return a validated interval for the optional Phase 8C CPU path."""

    for key in ("path", "reserve_rows", "gather_threads"):
        if start[key] != end[key]:
            raise ValueError("CPU-staging configuration changed: {}".format(key))
    counters = {}
    for key in (
        "batches", "cpu_rows", "payload_bytes", "index_bytes",
        "reallocations", "index_copy_ns", "host_gather_ns", "h2d_ns",
        "scatter_ns",
    ):
        delta = int(end[key]) - int(start[key])
        if delta < 0:
            raise ValueError("CPU-staging counter decreased: {}".format(key))
        counters[key] = delta
    if start["path"] == "mapped" and any(
        counters[key] for key in (
            "batches", "cpu_rows", "payload_bytes", "index_bytes",
            "index_copy_ns", "host_gather_ns", "h2d_ns", "scatter_ns",
        )
    ):
        raise ValueError("mapped CPU path unexpectedly reported staging work")
    return {
        "path": end["path"],
        "reserve_rows": int(end["reserve_rows"]),
        "gather_threads": int(end["gather_threads"]),
        **counters,
        "payload_gib": counters["payload_bytes"] / (1024.0 ** 3),
        "index_copy_seconds": counters["index_copy_ns"] / 1e9,
        "host_gather_seconds": counters["host_gather_ns"] / 1e9,
        "h2d_seconds": counters["h2d_ns"] / 1e9,
        "scatter_seconds": counters["scatter_ns"] / 1e9,
        "reconciled": True,
    }


def mixed_io_interval(start, end):
    """Return a reconciled interval for versioned mixed-I/O geometry."""

    if bool(start.get("enabled")) != bool(end.get("enabled")):
        raise ValueError("mixed-I/O mode changed within an interval")
    if not bool(end.get("enabled")):
        return {"enabled": False, "reconciled": True}
    legacy_geometry = {
        "cache_slot_bytes": 8192,
        "minimum_transfer_bytes": 4096,
        "subrow_count": 2,
        "feature_row_bytes": 4096,
    }
    start_geometry = start.get("geometry", legacy_geometry)
    geometry = end.get("geometry", legacy_geometry)
    if start_geometry != geometry:
        raise ValueError("mixed-I/O geometry changed within an interval")
    keys = (
        "group_rows", "raw_rows", "group_full_commands",
        "group_partial_commands", "raw_commands", "group_bytes",
        "raw_bytes", "group_part_hits", "raw_part_hits",
        "group_part_misses", "raw_part_misses", "physical_bytes",
    )
    result = {"enabled": True}
    for key in keys:
        value = int(end[key]) - int(start[key])
        if value < 0:
            raise ValueError("mixed-I/O counter decreased: {}".format(key))
        result[key] = value
    result["commands"] = (
        result["group_full_commands"]
        + result["group_partial_commands"]
        + result["raw_commands"]
    )
    expected_group_bytes = (
        result["group_full_commands"] * int(geometry["cache_slot_bytes"])
        + result["group_partial_commands"] * int(geometry["minimum_transfer_bytes"])
    )
    if result["group_bytes"] != expected_group_bytes:
        raise ValueError("mixed-I/O group bytes do not reconcile")
    if result["raw_bytes"] != (
        result["raw_commands"] * int(geometry["minimum_transfer_bytes"])
    ):
        raise ValueError("mixed-I/O raw bytes do not reconcile")
    if result["physical_bytes"] != result["group_bytes"] + result["raw_bytes"]:
        raise ValueError("mixed-I/O physical bytes do not reconcile")
    if result["group_part_hits"] + result["group_part_misses"] != (
        int(geometry["subrow_count"]) * result["group_rows"]
    ):
        raise ValueError("mixed-I/O group parts do not reconcile")
    raw_subrows = max(
        1,
        (int(geometry["feature_row_bytes"]) + int(geometry["minimum_transfer_bytes"]) - 1)
        // int(geometry["minimum_transfer_bytes"]),
    )
    if result["raw_part_hits"] + result["raw_part_misses"] != (
        raw_subrows * result["raw_rows"]
    ):
        raise ValueError("mixed-I/O raw parts do not reconcile")
    result["average_command_bytes"] = (
        result["physical_bytes"] / result["commands"]
        if result["commands"] else 0.0
    )
    result["physical_gib"] = result["physical_bytes"] / (1024.0 ** 3)
    result["geometry"] = dict(geometry)
    result["reconciled"] = True
    return result


def device_io_interval(start, end, *, maxima_scope="reset_interval"):
    """Validate a Phase 8E BaM SQ/CQ device-I/O measurement interval."""

    if maxima_scope not in ("reset_interval", "epoch_cumulative"):
        raise ValueError("unknown device-I/O maxima scope")
    if bool(start.get("enabled")) != bool(end.get("enabled")):
        raise ValueError("device-I/O statistics mode changed within an interval")
    if not bool(end.get("enabled")):
        return {"enabled": False, "reconciled": True}
    if int(start.get("outstanding", 0)) or int(end.get("outstanding", 0)):
        raise ValueError("device-I/O interval boundary has outstanding commands")
    additive = (
        "submitted_commands", "completed_commands", "completed_bytes",
        "active_ns", "total_latency_ns", "replay_commands", "replay_bytes",
    )
    result = {"enabled": True}
    for key in additive:
        value = int(end[key]) - int(start[key])
        if value < 0:
            raise ValueError("device-I/O counter decreased: {}".format(key))
        result[key] = value
    if maxima_scope == "reset_interval":
        # Existing epoch callers require a reset; preserve this strict contract.
        if int(start.get("max_latency_ns", 0)) != 0:
            raise ValueError("device-I/O latency maximum was not reset")
        if int(start.get("max_outstanding", 0)) != 0:
            raise ValueError("device-I/O outstanding maximum was not reset")
        result["max_latency_ns"] = int(end["max_latency_ns"])
        result["max_outstanding"] = int(end["max_outstanding"])
    else:
        # Maxima cannot be differenced. Validate monotonic epoch snapshots and
        # expose explicit epoch-prefixed fields, never claim batch-local maxima.
        for name in ("max_latency_ns", "max_outstanding"):
            before, after = int(start[name]), int(end[name])
            if before < 0 or after < before:
                raise ValueError("device-I/O cumulative maximum decreased: " + name)
            result["epoch_" + name] = after
        result["maxima_scope"] = "epoch_cumulative"
    if result["submitted_commands"] != result["completed_commands"]:
        raise ValueError("device-I/O submissions and completions do not reconcile")
    if result["replay_commands"] > result["completed_commands"]:
        raise ValueError("device-I/O replay commands exceed completions")
    if result["replay_bytes"] > result["completed_bytes"]:
        raise ValueError("device-I/O replay bytes exceed completed bytes")
    commands = result["completed_commands"]
    active_ns = result["active_ns"]
    result["primary_commands"] = commands - result["replay_commands"]
    result["primary_bytes"] = result["completed_bytes"] - result["replay_bytes"]
    result["average_command_bytes"] = (
        result["completed_bytes"] / commands if commands else 0.0
    )
    result["average_latency_ns"] = (
        result["total_latency_ns"] / commands if commands else 0.0
    )
    result["active_seconds"] = active_ns / 1e9
    result["device_gbps"] = (
        result["completed_bytes"] / active_ns if active_ns else 0.0
    )
    result["device_gib_per_second"] = (
        result["completed_bytes"] / (1024.0 ** 3) / (active_ns / 1e9)
        if active_ns else 0.0
    )
    result["completed_gib"] = result["completed_bytes"] / (1024.0 ** 3)
    result["reconciled"] = True
    return result


def reconcile_mixed_io_counters(native_io, gpu_cache, mixed_io, device_io):
    """Reconcile cache-fill events separately from physical NVMe commands.

    BaM counts one native miss for every active lane in the feature-row warp,
    so ``misses / 32`` is the number of cache insert/partial-fill events.  A
    partial fill can issue several minimum-transfer commands when the slot has
    more than two subrows; consequently that event count is not, in general,
    the physical command count.
    """

    if not mixed_io.get("enabled"):
        return {"enabled": False, "reconciled": True}

    native_misses = int(native_io["misses"])
    if native_misses % 32:
        raise ValueError("mixed-I/O native miss counter is not warp-aligned")
    native_fill_events = native_misses // 32
    cache_fill_events = (
        int(gpu_cache["inserts"]) + int(gpu_cache["partial_fills"])
    )
    if native_fill_events != cache_fill_events:
        raise ValueError(
            "mixed-I/O native fill events do not reconcile with cache fills: "
            "{} != {}".format(native_fill_events, cache_fill_events)
        )

    commands = int(mixed_io["commands"])
    subrow_count = int(mixed_io["geometry"]["subrow_count"])
    minimum_commands = cache_fill_events
    maximum_commands = cache_fill_events * subrow_count
    if not minimum_commands <= commands <= maximum_commands:
        raise ValueError(
            "mixed-I/O command count is outside the cache-fill geometry "
            "bounds: {} not in [{}, {}]".format(
                commands, minimum_commands, maximum_commands
            )
        )

    if device_io.get("enabled"):
        if int(device_io["primary_commands"]) != commands:
            raise ValueError(
                "mixed-I/O commands do not reconcile with device primary "
                "commands: {} != {}".format(
                    commands, int(device_io["primary_commands"])
                )
            )
        if int(device_io["primary_bytes"]) != int(mixed_io["physical_bytes"]):
            raise ValueError(
                "mixed-I/O bytes do not reconcile with device primary bytes: "
                "{} != {}".format(
                    int(mixed_io["physical_bytes"]),
                    int(device_io["primary_bytes"]),
                )
            )

    return {
        "enabled": True,
        "native_fill_events": native_fill_events,
        "cache_fill_events": cache_fill_events,
        "physical_commands": commands,
        "extra_commands_from_multi_subrow_fills": commands - cache_fill_events,
        "maximum_commands_from_geometry": maximum_commands,
        "device_counters_checked": bool(device_io.get("enabled")),
        "reconciled": True,
    }


def write_report(report, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = report.get("run_name") or "gids_baseline"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    report_path = output_dir / f"{run_name}-{timestamp}.json"
    jsonl_path = output_dir / "runs.jsonl"
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True)

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")

    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid and sudo_gid:
        for path in (output_dir, report_path, jsonl_path):
            try:
                os.chown(path, int(sudo_uid), int(sudo_gid))
            except OSError:
                pass
    return report_path, jsonl_path
