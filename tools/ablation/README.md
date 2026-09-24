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

At this publication, CPU command tests, source-snapshot integrity and native-module imports passed. Administrator installation, the real AE-account control check and this new service's native run are **pending**; existing accepted author measurements above do not certify the new command. After installation, use status/results to distinguish RUNNING, FINALIZING, PASS and FAILED. No final table is shown for a partial/failed request; old successes are not substituted. The installer starts one fresh acceptance request as `atc27_ae` after its namespace/input and permission checks succeed.

The administrator stages a root-owned copy of the exact author runtime manifest closure, binds the prepared graph/layout and SSD receipts read-only, and retains the original namespace paths inside the service. Snapshot SHA256: `9a10a2bae3a21220ff87cca8a577957745e44389ee92b0d156ef46e00df97e36`. The snapshot is a deployment artifact, not an included dataset, binary distribution, or claim of standalone rebuild acceptance. The previous CLI is preserved as `legacy_cli.py` by the administrator installer.

## Deployment marker repair

The initial server installation and reviewer launch succeeded, but the first GIDS smoke worker failed before training because the copied snapshot omitted `.digit-root`. The earlier nested-directory import test inherited the author root marker and did not prove isolation. A regression outside the author tree reproduced the failure and passed after adding the marker, checking 18 module origins without initializing CUDA. The controller now checks imports in a fresh isolated child inside the real service namespace before either selftest or native workers. The repair adds the marker to the snapshot manifest; training candidates and native binaries are unchanged. The server repair and restarted native acceptance remain pending.
