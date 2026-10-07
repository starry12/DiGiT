# Source map

| Component | Entry / source |
|---|---|
| Prepared-server reviewer commands | `tools/reviewer/cli.py` |
| Main source-build interface and integrity | `artifact.py`, `run.sh`, `artifact_integrity.py` |
| PA GraphSAGE training and sampler | `training/sage/` |
| PA GCN / GAT training | `training/gcn/`, `training/gat/` |
| Main evaluation and monitoring | `evaluation/`, `ae/pa_sage/` |
| GIDS loader and model definitions | `ae/papers/runtime/` |
| I/O accounting and native feature store | `runtime/io/` |
| PA ablation service | `tools/ablation/` |
| PA layout-grid service and dependencies | `tools/layout_grid/` |
| IG comparison service and dependencies | `tools/ig_performance/` |
| UKS comparison service and dependencies | `tools/uks_performance/` |
| UKL / CL comparisons | `tools/ukl_performance/`, `tools/cl_performance/` |
| Shared large-graph runtime | `tools/large_graph/runtime_sources/` |
| IG/UKS/UKL/CL GCN and GAT adapters | `tools/performance_models/` |
| Automatic GPU selection | `tools/gpu_selection/` |
| BaM dependency | `third_party/bam/` |

Use the [reviewer commands](REVIEWER.md) on the prepared server. The frontend dispatches directly to the selected experiment handler; service controllers enforce fixed protocols, data bindings and resource ownership. Reviewers do not execute the administrator scripts or research launchers.

The main source-build path has passed a clean native build and paired smoke for SAGE, GCN and GAT: [receipt](../reference/rebuilt_native_smoke.json). The supplementary service source targets separately provisioned read-only runtime snapshots; it is not a standalone supplementary build recipe for a new machine.

Supplementary `runtime_sources` directories preserve imported modules and source identities needed by the accepted runtime. Version-bearing dependency names are internal compatibility identifiers, not selectable alternative reviewer implementations. Their hash-bound verification chains must remain intact. Prepared data, native binaries and raw experiment logs are supplied separately on the server.

`reference/` contains selected accepted result records and required correctness constants. `provenance/` records source/data identities. Source mappings and preparation hashes support verification; reviewers can reproduce through the frontend without navigating those files.
