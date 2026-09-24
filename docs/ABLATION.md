# PA/SAGE supplementary component ablation

The prepared-server four-arm AE request **passed** on 2026-09-25 at 00:55:23 UTC+8. It ran from 2026-09-24 23:55:55 for 59 min 28 s, including smoke, initialization and full workers. All four smoke and four full workers were fresh processes, exited normally and recorded zero monitor query errors. Each full arm completed one seed-0 epoch: 1,207,179 examples and 1,179 updates, without validation/test. The service exited successfully and GPU 2 was released.

| Stage | Training time (s) | Speedup vs GIDS |
|---|---:|---:|
| GIDS | 194.63 | 1.0000× |
| +GR (adjacency only) | 196.12 | 0.9924× |
| ++NS | 160.38 | 1.2136× |
| DiGiT | 103.69 | 1.8770× |

All arms use a 4 GiB GPU feature cache and 11,105,992 CPU-cached rows. The first three share the main GIDS RevPR logical hot set. **+GR changes only adjacency order**: node IDs, edge multiset/EIDs, original feature payload, CPU cache lookup and legacy GPU replacement are retained. Its 196.12 s measurement is 0.77% slower than GIDS 194.63 s in this single observation.

++NS retains grouped feature storage, group-aware sampling and mixed I/O, with row-exact mapping of the same logical RevPR hot set. GR→NS therefore includes a feature-layout change. DiGiT changes CPU hot-set selection and GPU replacement together; NS→DiGiT is not FIFO alone. NS and DiGiT have identical losses, final model and batch shapes, with 49 fanout shortfalls out of 335,564,917 target edges each.

DiGiT takes **103.69 s (1.8770×)** against the measured GIDS baseline. The main PA/SAGE **1.8027×** remains a separate 20-epoch accuracy result. Training includes root-order generation, sampling, feature fetch and model updates; preparation, setup and teardown are separate. NS/DiGiT setup took 505.02/363.79 s. These single-seed first-epoch measurements make no steady-state, variance or accuracy claim.

The [full-precision CSV](../reference/pa_sage_ablation_perf.csv) includes setup and observed monitoring delays. The [acceptance receipt](../reference/pa_sage_ablation_perf.json) records protocols, per-arm I/O, all eight workers and 85 independently rechecked evidence hashes. The [source index](../provenance/pa_sage_ablation_perf_manifest.json) binds candidate, deployed controller, monitor and admission-policy identities. Private input bindings, datasets and checkpoints are not included in the public package.

## Earlier measurements

The accepted afternoon author results remain unchanged in the [historical author receipt](../reference/pa_sage_ablation_author_20260924.json) and [CSV](../reference/pa_sage_ablation_author_20260924.csv): 194.45/194.76/161.73/104.50 s, 1.8608×. That record retains its publication-time extension status; it is not the current AE status. Its [source index](../provenance/pa_sage_ablation_author_20260924_manifest.json) is preserved byte for byte.

The [older cache-disabled receipt](../reference/pa_sage_ablation_perf_v2_historical.json) retains 227.37/167.93/181.51/105.60 s. The old-layout GR 151.75 s and failed NS 162.05 s observations remain excluded from the selected table. Prior incomplete AE attempts are retained on the server; no failed attempt is relabeled as accepted.

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

## Current AE reproduction status — 2026-09-25

The prepared-server four-arm AE request **passed** on 2026-09-25 at 00:55:23 UTC+8. It ran from 2026-09-24 23:55:55 for 59 min 28 s, including smoke, initialization and full workers. All four smoke and four full workers were fresh processes, exited normally and recorded zero monitor query errors. Each full arm completed one seed-0 epoch: 1,207,179 examples and 1,179 updates, without validation/test. The service exited successfully and GPU 2 was released.

Accepted request: `20260924_235555_5afe836d9f74`. All eight workers and their monitoring belong to this request; no accepted-smoke reuse claim was used.

## Monitoring and resource admission

The service uses the original author external `nvidia-smi` monitor: a 5-second subprocess query timeout and a 0.5-second wait after each query. The first successful sample is required before a worker starts. Later query errors are recorded while sampling continues; final acceptance requires zero query errors, normal worker and monitor exits, valid ownership, and resource/report checks. There is no live heartbeat or phase-gap gate. Available-host-memory admission is 192 GiB; cache capacities and training candidates are unchanged.

Earlier NVML attempts stopped on monitoring latency. The successful request uses the restored `nvidia-smi` implementation; its zero-error record does not establish the root cause of those earlier delays.
