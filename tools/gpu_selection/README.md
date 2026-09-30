# Automatic GPU selection for prepared-server AE requests

Installed for PA main models (SAGE/GCN/GAT), PA/SAGE ablation, the 15-point PA/SAGE
layout grid and IG/SAGE performance. Existing reviewer commands need no GPU option.
UKS retains its separately installed automatic selection implementation.

The service samples GPUs 0–3 twice, chooses an idle NVIDIA L40, takes a per-GPU
lock shared with UKS, and rechecks occupancy. The whole request keeps one physical
GPU and UUID, including all workers and monitoring. No suitable GPU causes a failed
request, not a queued run or a mid-request switch. Global experiment and SSD locks
remain in force. Selection requires at least 40 GiB free, at most 5% utilization,
and no compute process; PA main also preserves its 256 MiB used-memory ceiling,
while the other three retain a used-memory limit below 1 GiB.

`digit-ae status` displays the selected GPU. Machine-readable evidence retains its
UUID and the transport hash. IG CPU2 affinity, workloads, timing and compute kernels
are unchanged. The explicit IG controller and telemetry adapters route the selected
GPU while retaining the frozen training source and its separate identity.

## Source and deployment

[deployment.json](deployment.json) maps every published controller, CLI, service
unit and shared adapter to its installed path and SHA256. Published code matches
the installed files byte for byte. Existing extension controllers live in
[ablation](../ablation/), [layout_grid](../layout_grid/) and
[IG performance](../ig_performance/service/); the main-model controller is under
[service](service/selfservice_controller.py).

These are administrator-managed prepared-server components, not a standalone
installer for a new machine. They require the existing root-owned runtime snapshots,
Python environment, data mounts, device access and fixed service permissions.
Deploy all mapped files together while no request is running, preserve ownership
and modes, and reload systemd after unit changes. The server installation already
backed up the previous files and passed isolated CPU checks in all four runtime
namespaces. Reviewers should use the ordinary `digit-ae` commands, not deploy files.

## Validation

```bash
python3 -B tools/gpu_selection/test_selection.py
```

The 14 tests cover simulated card selection, occupancy races, held locks, device
identity, monitoring identity and source syntax; they do not launch training.
[Installation evidence](../../reference/automatic_gpu_installation.json) records
CPU-only namespace checks simulating all four GPU indices without CUDA initialization.
Fresh native acceptance for these four updated entry points remains pending; older
accepted experiment results remain historical evidence. UKS native acceptance is
recorded separately in its own result receipt.
