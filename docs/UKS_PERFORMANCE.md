# UKS/SAGE Freq + BFS performance supplement

| Reference | Maximum same-round speedup over five rounds |
|---|---:|
| Author native run, 2026-09-30 | **1.75×** |

All ten author performance workers passed and exited normally. This is not yet an
AE-account acceptance. The service is installed separately from PA/IG and preserves
`ae-pa-v1`. Installation performs only a CPU namespace/import check; a subsequent
reviewer request is needed for native AE acceptance.

### UKS / GraphSAGE performance: Freq + BFS

The UKS extension packages five paired performance windows with real sampling,
SSD I/O and model updates. **Server installation and CPU namespace/import checks passed; a fresh AE replay is pending**;
the currently reported result is an author reference.
To start an independent request:

```bash
digit-ae performance UKS sage
digit-ae status UKS sage --action performance
digit-ae results UKS sage --action performance
digit-ae logs UKS sage --action performance
```

Each system runs 20 warmup + 300 timed mini-batches per round. Results display the
maximum same-round GIDS/DiGiT speedup out of five rounds. GIDS uses its original
random window and RevPR CPU heat; DiGiT uses a BFS window and independently
presampled Freq CPU heat. **The node windows differ.** This measures the combined
configuration and workload choices, not an isolated caching or I/O improvement.
Use `digit-ae results UKS sage --action performance --reference` for the
`AUTHOR_REFERENCE`, or `digit-ae stop UKS sage --action performance` to cancel.
[UKS protocol and source](UKS_PERFORMANCE.md).

## Fixed protocol

- UKS: 133,633,040 nodes; normalized directed CSC keeps nonself multiedges and
  adds one self-loop per node, without reverse-edge insertion.
- Logical-node-consistent 256D FP32 synthetic features and synthetic 19-class
  labels; replicas have identical features. No accuracy or convergence claim.
- GraphSAGE: 3 layers, hidden 128, fanouts 10/5/5, batch 1024, seed 0,
  dropout 0.2, Adam lr 0.001 and weight decay 0.001.
- Both: 13,363,304 CPU cached logical nodes (~12.74 GiB feature bytes),
  4 GiB GPU feature cache, **1 KiB cache lines**, and 16 MiB DMA scratch.
  Metadata and input hash tables are additional to feature-cache capacity.
- GIDS: RevPR CPU heat, legacy GPU cache, old random root window, default CPU
  scheduling. DiGiT: independent 100-batch seed-23 Freq CPU heat, GPU FIFO,
  BFS window, logical CPU2, g2/r20 layout and incremental group sampling.
- DiGiT merges eligible jointly cold partners into 2 KiB SSD commands while
  retaining independent 1 KiB cache entries; unpaired/partially hit reads use 1 KiB.
  Both window-buffer and accumulator flags remain off for this UKS protocol.
- Freq reuses the accepted independent presampling result (no feature I/O or
  optimizer updates during presampling). The finite sample does not guarantee
  nonzero frequency for every Top-K cache entry.
- BFS covers the whole 13,363,304-node training set on its induced undirected
  graph. Components start at the highest-degree unseen node, ties by node ID;
  neighbors use ascending node IDs. Complete 1024-node blocks are shuffled with
  seed 0, epoch 0; the first 320 blocks form the window. Only 8,115 roots overlap
  the 327,680-root GIDS window. The BFS order is reused rather than rebuilt.
- Five rounds in G1,D1,G2,D2,G3,D3,G4,D4,G5,D5 order, fresh processes/caches/models.
  Each has 20 warmup and 300 measured optimizer updates. Report the maximum
  **same-round** GIDS seconds / DiGiT seconds, not independently selected times.
  Warmup, setup, presampling and BFS generation are excluded. No stable-speedup,
  interference-free, full-epoch or paper-exact equivalence claim is made.

## Prepared-server execution and evidence

The runtime reuses the accepted GIDS short-check evidence and runs a fresh DiGiT
adaptive short (4–64 updates, requiring CPU/GPU/SSD path coverage), then ten
performance workers. It checks sampled edges and bit-exact features in the short,
finite training values, update counts, cache/I/O reconciliation, report hashes,
within-arm root/hot identities, and normal exits. It never writes the raw SSD.
Author results retain all five pairs and the initial failed short-check attempt.

The service retains the existing AE/author GPU/SSD locks and automatically selects
one idle L40 among physical GPUs 0–3. Selection requires two idle samples,
at least 40 GiB free, GPU utilization at most 5%, and no compute process. It takes
a per-GPU lock and rechecks availability; all workers then use the same GPU UUID.
The default ranking is most free memory, then lowest physical index. Each worker
rechecks the selected GPU before CUDA initialization; it stops rather than changing
cards mid-request. Host admission remains 256 GiB available. These
are admission checks, not strict continuous resource monitoring. No automatic
retries are added. Requests continue after SSH disconnect; `PASS` requires service
success and revalidated complete evidence. Historical reference is never silently
substituted for an incomplete/failed latest request.

The public `tools/uks_performance/runtime_sources` preserves the source dependency
closure and hashes; `service` contains fixed reviewer adapters. Private binary,
input identity and historical receipt files are installed only in the root-owned
server snapshot. Prepared data on `/mnt/n0` are read-only to the service. This
extension targets the prepared server; a clean independent rebuild of this UKS
extension has not yet been validated. See `provenance/uks_sage_performance_sources.json`.

## Automatic GPU transport v2

The accepted author computation is frozen. A separate worker adapter replaces only
the GPU2-specific admission and invokes the same smoke/performance functions.
Cache policy, sampler, CPU affinity, training windows, model and result selection
are unchanged. The chosen physical index and UUID are saved per request and worker;
completion requires the same assignment and transport hash throughout. A GPU
reservation coordinates this launcher, but cannot prevent unrelated users from
starting their own workloads. The new GPU transport has CPU checks; a fresh AE
native run is still required. The 1.75× reference remains the original GPU2 run.

The installed automatic-GPU transport and reviewer CLI were verified without starting training. [Installation receipt](../reference/uks_sage_server_installation.json).
