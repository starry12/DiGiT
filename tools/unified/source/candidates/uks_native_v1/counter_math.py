"""Cumulative I/O snapshots: subtract additive counters, never subtract maxima.

When a window does not set a new record, its exact maximum is unknown. Keep the
cumulative upper bound and report the window maximum as None. A complete epoch
starts from reset counters, so its maxima remain exact.
"""

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


def device_io_interval(start, end, complete_region=False):
    """Validate a Phase 8E BaM SQ/CQ device-I/O measurement interval."""

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
    if complete_region and any(int(start.get(k,0)) for k in additive+("max_latency_ns","max_outstanding")):
        raise ValueError("complete epoch must start from reset device counters")
    for key in ("max_latency_ns","max_outstanding"):
        before=int(start.get(key,0));after=int(end[key])
        if before<0 or after<before:
            raise ValueError("device-I/O cumulative maximum decreased: "+key)
        if result['completed_commands']==0 and after!=before:
            raise ValueError("device-I/O maximum changed without completed commands")
        result['cumulative_'+key]=after
        # A new cumulative record was necessarily attained in this window.
        result[key]=0 if result['completed_commands']==0 else after if before==0 or after>before else None
        result[key+'_exact']=result[key] is not None
    result['maxima_scope']='complete_epoch_from_reset' if complete_region else 'window_exact_when_known_with_cumulative_upper_bounds'
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
