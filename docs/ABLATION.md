# PA/SAGE supplementary component ablation

Updated accepted author measurements on 2026-09-24. Each arm completed one entire seed-0 training epoch (1,207,179 examples; 1,179 updates), without validation/test. GIDS and revised +GR reuse accepted independent runs; ++NS and DiGiT are fresh runs completed at 18:55:18 UTC+8. Every selected worker exited normally with zero monitor query errors. All runs used physical GPU 2.

| Stage | CPU cache rows | Training (s) | Speedup vs measured GIDS |
|---|---:|---:|---:|
| GIDS (main RevPR CPU cache) | 11105992 | 194.45 | 1.0000x |
| +GR (adjacency only) | 11105992 | 194.76 | 0.9984x |
| ++NS | 11105992 | 161.73 | 1.2023x |
| DiGiT | 11105992 | 104.50 | 1.8608x |

The first three arms share the main GIDS RevPR logical hot set (11,105,992 rows); every arm has a 4 GiB GPU feature cache. **+GR changes only adjacency order**: node IDs, edge multiset/EIDs, original feature payload, CPU cache lookup and legacy GPU replacement are retained. Its 194.76 s measurement is essentially unchanged from GIDS 194.45 s in this single observation (+0.16% time).

++NS retains grouped feature storage, group-aware sampling and mixed I/O, with row-exact mapping of the same logical RevPR hot set. Thus GR→NS includes the feature-layout change and is not an isolated NS effect. DiGiT uses the frequency-selected CPU hot set and FIFO GPU replacement together; NS→DiGiT is not FIFO alone. NS and DiGiT have identical losses, final model and batch shapes; both record 49 fanout shortfalls out of 335,564,917 target edges. BFS remains disabled.

DiGiT takes **104.50 s (1.8608x)** against the measured 194.45 s GIDS baseline. The main PA/SAGE **1.8027x** result still describes the separate 20-epoch accuracy protocol; do not replace its denominator or claim these are repeated measurements of the same protocol. Training includes root-order generation, sampling, feature fetch and model updates, but excludes preparation and setup. NS/DiGiT setup costs are 505.39/435.55 s, separately recorded. No warmup, steady-state, multi-run variance or accuracy claim is made.

The previous published CPU-cache-disabled experiment (227.37 / 167.93 / 181.51 / 105.60 s, 2.1531x, NS 8.09% slower than GR) remains in the [historical receipt](../reference/pa_sage_ablation_perf_v2_historical.json). It uses a different cache/GR definition and is not the current table. The later old-layout GR 151.75 s result is also excluded: it changed feature storage rather than adjacency. The earlier NS 162.05 s attempt failed strict monitoring and is not the accepted NS value.

The [curated receipt](../reference/pa_sage_ablation_perf.json) records per-arm protocols, I/O, setup costs, source versions, acceptance and evidence hashes. The [source identity index](../provenance/pa_sage_ablation_perf_manifest.json) identifies the unchanged v3 NS/DiGiT runtime and revised v4 GR candidate. These are accepted author results; a separate prepared-server extension is supplied below; `run.sh` and the sealed main reviewer runtime are unchanged. The submitted `ae-pa-v1` and main three-model results remain separate.

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
