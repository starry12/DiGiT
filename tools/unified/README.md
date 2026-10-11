# Unified GraphSAGE experiments

Version `ae-20261011-v2` includes the current PA, IG, UKS, UKL and CL implementation, its dependencies, and the server controls. The original PA main and ablation entry points remain available. This directory supersedes the older large-dataset short-window performance entries for GraphSAGE.

## On the provided server

```bash
source /srv/digit-ae/activate.sh
digit-ae unified
digit-ae performance UKS sage --mode short1
digit-ae status UKS sage --action performance --mode short1
digit-ae logs UKS sage --action performance --mode short1
digit-ae results UKS sage --action performance --mode short1
```

Substitute `PA`, `IG`, `UKL` or `CL` for the dataset. `digit-ae unified` reports the installed availability and validation status. Publication of this source package alone does not install it on the server. The administrator installs the separately prepared, machine-specific update; the public `server/` directory contains the interface implementation. Installing that update does not start experiments.

| Mode | Training extent | Report |
|---|---|---|
| `short1` | One pair, 20 warm-up and 300 measured batches per arm | Single paired ratio and measured I/O/cache counters |
| `short5` (default) | Five pairs of the same short window, alternating arm execution order | Median of the five paired speedups, with every round shown |
| `full20` | 20 complete training epochs per arm | Ratio of total training times and per-epoch rows |

Only PA `full20` evaluates validation and final test accuracy. All other combinations report training performance and I/O, without an accuracy claim. A short window with GIDS random roots and DiGiT BFS roots is **not a matched-example comparison**; the full-epoch modes retain the complete training set. This release does not label the random/BFS experiment as the separate historical matched-BFS diagnostic.

New unified requests are serialized and use the shared experiment locks and GPU 2. Preparation for large graphs can dominate elapsed time. They do not inherit a previous request's PASS. A PASS requires the native paired report, successful completion and matching source bindings. Stop with `digit-ae stop UKS sage --action performance --mode short1`. Historical performance results remain inspectable with `--legacy`.

The printed output directory under `/srv/digit-ae/unified-results/` contains `summary.json`, `summary.csv`, configuration, progress, logs and completion records. For a failed request, read its error and controller log; a launch acknowledgement is not a successful experiment.

## Validation at publication

| Dataset | Available evidence | Remaining scope |
|---|---|---|
| PA | Completed BFS256 20-epoch pair with validation and final test | The newly installed wrapper has not been rerun |
| UKS | Completed short1 pair | No new five-pair or full20 acceptance claimed |
| IG | Native run completed; report accounting corrected offline | No fresh GPU replay of the accounting correction |
| UKL | DiGiT short1 completed successfully | Earlier GIDS computation exited unsuccessfully at finalization; complete paired validation pending |
| CL | Source and CPU checks; preparation in progress at publication | Prepared inputs and complete paired validation pending |

UKL and CL are intentionally available as **pending-validation** entries for reviewer execution. CL requires completion of its prepared-input contract before training can start; the entry fails with the missing preparation path if it is not ready. Neither availability nor publication claims that CL has passed. See [validation-status.json](validation-status.json) and [reference](reference/) for the precise evidence. The reports are historical records with their own hashes, not acceptance of every mode of the current global source manifest.

## Feature checklist

| Feature | GIDS | DiGiT |
|---|---|---|
| Training root order | Random | BFS chunks of 256 nodes, recombined each epoch |
| Neighbor sampling | Uniform | Strict group-first sampling with groups of two |
| CPU feature cache | Per-node RevPR | Frequency-selected cache; equal feature-byte capacity within each pair |
| Dynamic GPU feature cache | Legacy GIDS policy, window buffering disabled | FIFO policy, window buffering disabled; equal 4 GiB feature capacity |
| SSD requests | Single feature rows | Single rows and two-row grouped requests |
| Training and reporting | Same dataset/model/optimizer parameters within each pair | Same; training time, cache service fractions and actual SSD command/byte counters |
| Execution optimizations | Baseline path | Asynchronous feature path and model/CUDA execution optimizations; see the dataset adapter for capture boundaries |

Rows are 512 B for PA/UKL/CL, 1 KiB for UKS and 4 KiB for IG; paired requests are twice those sizes. Dataset-specific graph geometry, feature dimensions and labels are retained. UKS/UKL/CL use synthetic feature/label inputs and are not accuracy benchmarks. In IG/UKL/CL short modes, frequency profiling uses 320 batches and fills the remaining cache from the recorded fallback; this is a functional check, not evidence of full-epoch cache effectiveness. PA reuses the fixed parent full-epoch profile rather than reprofiling for BFS256. Detailed effective configuration is saved for each request.

## Source and deployment boundary

`source/` preserves the selected current working-tree code byte for byte; [source-files.json](source-files.json) binds every included file. Internal package names are retained because the current implementation imports them. Server binaries, historical dependency receipts, raw logs and machine-specific installation plans are excluded from this source addition; native integrity checks still require the corresponding server environment. Selected public reference summaries omit server-local paths and runtime identities; their original and archived hashes are recorded separately. The top-level project license covers author-owned code; retained BaM, LLVM/libc++ and NVIDIA notices continue to apply.

The provided server runs the bound `/home/embed/digit` tree with the existing Conda/CUDA/DGL environment, prepared data and SSD layout. Native workers retain server-specific paths and identity checks. This archive is not a validated standalone, relocatable rebuild, and does not include raw graphs, feature payloads, checkpoints or credentials. Follow the repository environment/native-build/data guides for the original artifact; additional unified native build scripts are included under `source/candidates/`. Do not execute a privileged server installer on another machine without adapting its reviewed paths and prepared inputs.

Verify the published package with `bash run.sh verify` at the repository root. To verify this snapshot's file hashes only, without initializing CUDA or touching the SSD, use `python3 tools/unified/verify_snapshot.py`.
