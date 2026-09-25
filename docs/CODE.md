# Code map

| Component | Location |
|---|---|
| Single public entry, package/deployment guards | `artifact.py`, `run.sh`, `artifact_integrity.py` |
| GraphSAGE training, selected protocol, native sampler | `training/sage/` |
| GCN and GAT math, budgets, workers | `training/gcn/`, `training/gat/` |
| SAGE orchestration, independent monitoring and acceptance | `evaluation/sage/`, `ae/pa_sage/` |
| GCN/GAT entry binding | `evaluation/gcn/`, `evaluation/gat/` |
| DiGiT layout and sampler support | `training/sage/runtime/digit/` |
| GIDS loader and common model definitions | `ae/papers/runtime/` |
| Effective I/O counters and native feature store | `runtime/io/` |
| BaM dependency | `third_party/bam/` |

The selected model protocol files are byte-identical to the preserved release. Python imports and filesystem paths were updated to this tree; the nested versioned dispatch layers were replaced by one direct entry. Native math and kernel source are unchanged apart from required paths. The instrumented BaM `page_cache.h` is retained in `runtime/io/native/include/` and takes precedence over the shared vendor headers.

`provenance/source_map.json` records source hashes, new file hashes and adaptations. `provenance/preparation_hashes.json` retains the identities bound by existing prepared-data receipts without carrying old preparation implementations. `reference/sage_correctness.json` contains only the values used by the live SAGE deterministic equivalence check.

No research Git history or old implementation snapshots are included. Data contract names and receipt schemas remain stable for compatibility. Source reorganization has CPU/static validation; native source rebuild and a fresh native pair from this tree remain separate acceptance work. The prepared service stays on its preserved release during ongoing experiments.

## Supplementary layout grid — 2026-09-26

The accepted author 15-point grid and exact supplementary code are now indexed in [LAYOUT_GRID](LAYOUT_GRID.md). The prepared-server command is `digit-ae layout PA sage` after extension installation. It produces fresh evidence; publication of the author result is not AE replay acceptance. The main experiments and immutable `ae-pa-v1` are unchanged.
