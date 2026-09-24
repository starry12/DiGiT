# Prepared-server four-arm extension

## Four-arm AE command

The prepared-server extension adds these commands after administrator installation:

```bash
digit-ae ablation PA sage
digit-ae status PA sage --action ablation
digit-ae results PA sage --action ablation
digit-ae logs PA sage --action ablation
digit-ae stop PA sage --action ablation
```

Each request runs four fresh smoke workers, then GIDS → adjacency-only +GR → ++NS → DiGiT for one complete training epoch each, without validation/test. It reuses prepared data but does not reuse historical performance reports. GPU 2, data locations, units and protocol are fixed; busy requests fail instead of queueing. The global GPU/NVMe lock is shared with the main AE services. Starting and stopping require only two exact administrator-installed sudo permissions; no arbitrary privileged command, configuration, GPU or output path is accepted.

The service uses a separate root-owned snapshot and read-only prepared-data mounts. Existing PA/IG commands and the submitted sealed release remain separate. The extension sources are under `tools/ablation/`; they target this prepared server and its matching administrator-provisioned runtime snapshot, not a standalone ablation run from the concise checkout. The standard two-arm `run.sh` interface remains unchanged.

## Monitoring and resource admission

The updated four-arm controller uses the same external `nvidia-smi` monitor sources as the accepted afternoon author runs. Each query has a 5-second subprocess timeout, followed by a 0.5-second wait. The first successful sample is required before a worker starts. Later query errors are recorded while sampling continues; final acceptance requires zero query errors, a normally exited worker and monitor, valid ownership, and resource/report checks. The later NVML live heartbeat and phase-gap gates are removed. The host admission threshold stays at 192 GiB; cache sizes and training candidates are unchanged.

The source update is prepared and tested; administrator installation of this monitoring change is pending. The previously installed service still uses the recorded NVML policy until that installation succeeds. Retesting remains paused. The installer for this update performs a namespace selftest only and does not launch training. Monitor/source checks alone do not establish native acceptance.

The original modules are [`gpu_monitor.py`](../../ae/pa_sage/gpu_monitor.py) and [`monitor_control.py`](../../ae/pa_sage/monitor_control.py). CPU fault injection is available with `python3 -B tools/ablation/monitor_tests.py`; fixed-command and result validation tests use `python3 -B tools/ablation/tests.py`. Neither command starts GPU training.

## Current server state

The extension, root-marker repair, isolated runtime imports and 192 GiB admission policy have already been installed. The most recent AE request reused accepted GIDS/+GR smoke and completed ++NS smoke. DiGiT smoke stopped when a successful NVML query took 13.03 seconds and exceeded its then-active 5.5-second heartbeat rule. Four full epochs remain pending. The accepted author table is separate from this incomplete AE reproduction.

The controller defaults to four fresh smoke workers followed by four fresh full epochs. An administrator may provision a hash-bound, one-time accepted-smoke reuse claim; `ablation_resume.py` validates it before use. The previous claim is already consumed, and this monitoring update neither resets it nor requests another experiment. No historical full-epoch performance report is reused.

## Runtime snapshot

The fixed worker launcher applies the 192 GiB admission policy to the administrator-provisioned candidate snapshot. The candidate manifests, cache configurations, binaries and prepared data retain their identities. The runtime snapshot SHA256 remains `9a10a2bae3a21220ff87cca8a577957745e44389ee92b0d156ef46e00df97e36`. This is a prepared-server deployment artifact; the concise checkout does not include the dataset or private input bindings. The submitted `ae-pa-v1` and main two-arm services remain separate.
