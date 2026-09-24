# Prepared-server four-arm extension

The prepared-server four-arm AE request **passed** on 2026-09-25 at 00:55:23 UTC+8. It ran from 2026-09-24 23:55:55 for 59 min 28 s, including smoke, initialization and full workers. All four smoke and four full workers were fresh processes, exited normally and recorded zero monitor query errors. Each full arm completed one seed-0 epoch: 1,207,179 examples and 1,179 updates, without validation/test. The service exited successfully and GPU 2 was released.

## Four-arm AE command

The extension is installed on the prepared server. To read existing accepted results without starting another run, use `digit-ae results PA sage --action ablation`.

```bash
digit-ae ablation PA sage
digit-ae status PA sage --action ablation
digit-ae results PA sage --action ablation
digit-ae logs PA sage --action ablation
digit-ae stop PA sage --action ablation
```

A new request runs fresh GIDS → adjacency-only +GR → ++NS → DiGiT smoke workers, then one full training epoch per arm. GPU 2, data paths and protocol are fixed. Requests share the main AE GPU/NVMe exclusion lock and reject busy resources. The service survives SSH disconnects. `PASS` requires closed accepted reports and successful service exit; the CLI never substitutes an earlier success for a later failure.

The extension source is in `tools/ablation/` and targets a separately provisioned runtime snapshot with read-only data mounts. It does not provide a standalone four-arm run from the concise checkout. The main two-arm `run.sh` interface and submitted `ae-pa-v1` remain separate.

## Monitoring and resource admission

The service uses the original author external `nvidia-smi` monitor: a 5-second subprocess query timeout and a 0.5-second wait after each query. The first successful sample is required before a worker starts. Later query errors are recorded while sampling continues; final acceptance requires zero query errors, normal worker and monitor exits, valid ownership, and resource/report checks. There is no live heartbeat or phase-gap gate. Available-host-memory admission is 192 GiB; cache capacities and training candidates are unchanged.

The original modules are [`gpu_monitor.py`](../../ae/pa_sage/gpu_monitor.py) and [`monitor_control.py`](../../ae/pa_sage/monitor_control.py). CPU checks use `python3 -B tools/ablation/monitor_tests.py` and `python3 -B tools/ablation/tests.py`.

## Deployment and evidence

The [accepted receipt](../../reference/pa_sage_ablation_perf.json) and [source index](../../provenance/pa_sage_ablation_perf_manifest.json) identify the completed request, runtime snapshot, controller, worker launcher and admission wrapper. The snapshot SHA256 remains `9a10a2bae3a21220ff87cca8a577957745e44389ee92b0d156ef46e00df97e36`.

All eight workers in the selected request were fresh. The controller supports an administrator-provisioned, hash-bound one-time smoke reuse claim, but the earlier claim was already consumed and was not used here. Full performance reports are never reused. Private bindings and native binaries are part of the separate prepared-server deployment.
