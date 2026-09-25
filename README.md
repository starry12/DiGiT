# DiGiT

DiGiT is an out-of-core graph neural network training system built on GIDS. It combines GPU-side neighbor sampling, SSD-backed feature access and multilevel caching to train graphs whose features exceed GPU memory.

This repository provides the DiGiT implementation, the GIDS comparison path, and experiment entry points for **GraphSAGE, GCN and GAT**. Measurements cover training time, accuracy and effective I/O. See the [code map](docs/CODE.md) for the implementation and [scope and claims](docs/CLAIMS.md) for the current evaluation coverage.

## Getting started

| What you want to do | Start here |
|---|---|
| Run experiments on the provided AE server | [Main experiments, ablation and layout-grid commands](#ae-environment-and-reproduction) |
| Create an environment from scratch on your own machine | [Environment setup](docs/ENVIRONMENT.md): prerequisites, locked dependencies and checks |
| Compile DiGiT and its GIDS/BaM dependencies | [Native build guide](docs/NATIVE_BUILD.md) |
| Connect the prepared graph, features and SSD data | [Data guide](docs/DATA.md) |

## AE environment and reproduction

The provided AE server includes the Python/CUDA environment, compiled native components, prepared read-only data, and the installed ablation and layout-grid extensions. Log in with the reviewer SSH account supplied privately through AE, then activate the environment:

```bash
source /srv/digit-ae/activate.sh
```

**Run one request at a time:** main models, ablation and the layout grid share GPU 2 and the SSD. Wait for the current request to finish and release resources before starting the next; busy requests are rejected rather than queued. Each launch creates fresh results and continues after SSH disconnects. No local build or data preparation is needed on this server.

### Main experiments: PA × GraphSAGE / GCN / GAT

Choose one model below. Each `run` request performs paired smoke first, then trains **GIDS → DiGiT for 20 epochs each**, with full validation after every epoch and one final test per system.

| Model | Start a new full comparison | Read its results |
|---|---|---|
| GraphSAGE | `digit-ae run PA sage` | `digit-ae results PA sage --action run` |
| GCN | `digit-ae run PA gcn` | `digit-ae results PA gcn --action run` |
| GAT | `digit-ae run PA gat` | `digit-ae results PA gat --action run` |

For progress and recent logs, use the same model name:

```bash
digit-ae status PA sage --action run
digit-ae logs PA sage --action run
```

For a short check on its own, use `digit-ae smoke PA sage`, then `digit-ae results PA sage --action smoke`; substitute `gcn` or `gat` as needed. A full `run` already includes smoke, so this separate request is optional.

### Component ablation: PA / GraphSAGE

This request runs four-arm smoke, then **GIDS → +GR (adjacency only) → ++NS → DiGiT**, each for **one complete training epoch (1,179 updates), without validation/test**:

```bash
digit-ae ablation PA sage
digit-ae status PA sage --action ablation
digit-ae results PA sage --action ablation
digit-ae logs PA sage --action ablation
```

Expect a four-row table of training seconds and speedup versus GIDS after acceptance. The existing AE run has passed; [protocol and evidence](docs/ABLATION.md) describe the cache settings and component boundaries.

### Grouping and replication grid: PA / GraphSAGE

This request covers **group sizes {1, 2, 4} × replication ratios {0%, 10%, 20%, 40%, 80%}**, for **15 fresh full-epoch measurements (1,179 updates each), without validation/test**:

```bash
digit-ae layout PA sage
digit-ae status PA sage --action layout
digit-ae results PA sage --action layout
digit-ae logs PA sage --action layout
```

The service reuses the 15 prepared layouts and shared proxy-feature SSD region, with real sampling, I/O and model updates. It runs a fresh g2/r20 smoke before the full grid; the other 14 points retain the declared runtime checks. It does not regenerate large layouts. Status reports completion out of 15; accepted results list training seconds and speedup relative to g2/r20.

The published author grid is accepted and the AE extension is installed; a fresh AE grid replay has not yet been accepted. To view the author measurements without starting a run:

```bash
digit-ae results PA sage --action layout --reference
```

This explicitly prints `AUTHOR_REFERENCE`, separate from a new AE request's `PASS`. See [grid protocol and evidence](docs/LAYOUT_GRID.md) for proxy-feature calibration and verification scope.

### Completion, stopping and result files

`status`, `logs` and `results` only inspect records. Use the matching `--action run`, `--action ablation` or `--action layout` above to select the experiment. They select the latest request for that action, including a failed one. Final **`PASS` requires accepted reports and successful service completion**; a launch acknowledgement or running/provisional table is not final acceptance. The printed output path contains the logs, reports and summaries and can be downloaded with SFTP/SCP through the supplied SSH route.

To cancel a request, use its matching command and wait for inactive status and resource release:

| Request | Stop command |
|---|---|
| Main comparison | `digit-ae stop PA sage` (substitute `gcn` or `gat`) |
| Four-arm ablation | `digit-ae stop PA sage --action ablation` |
| Layout grid | `digit-ae stop PA sage --action layout` |

The main SAGE full workflow previously took about **2 h 25 min**, and four-arm ablation about **59 min**, including smoke and setup. These are observed wall times, not estimates from the per-epoch result tables; server load can change them. [Reviewer instructions](docs/REVIEWER.md) provide additional troubleshooting and environment details.

The main experiments execute the preserved, accepted server release; the supplementary services use separately installed runtime snapshots. This repository reorganizes the main release into a concise source tree; its native compilation in a fresh locked environment and subsequent PA/SAGE, GCN and GAT preflight and paired smoke have passed; see the [rebuilt-source receipt](reference/rebuilt_native_smoke.json). Source origins and adaptations are recorded in [the source map](provenance/source_map.json). Server runtimes are updated through explicit installation.

## Current evaluation scope

The current artifact evaluates **Papers100M (PA)** with GraphSAGE, GCN and GAT, comparing GIDS and DiGiT. Each main comparison uses seed 0, 20 epochs, full validation and one final test. The supplementary PA/SAGE ablation and grouping/replication grid use one complete epoch per setting and report performance only. This is a reconstruction of the paper implementation.

The source tree contains one selected implementation per model. It excludes research Git history, intermediate implementations, training logs, checkpoints, compiled binaries and datasets.

## Results and boundaries

| Model | Selected training speedup | GIDS test accuracy | DiGiT test accuracy | Evidence |
|---|---:|---:|---:|---|
| GraphSAGE | 1.8027× | 62.8531% | 62.9552% | Fresh reviewer full pair; strict monitoring passed |
| GCN | 1.8584× | 53.7422% | 53.3634% | Fresh reviewer full pair; strict monitoring passed |
| GAT | 1.7266× | 48.6204% | 47.9574% | Fresh reviewer full pair; strict monitoring passed |

These selected measurements are [fresh reviewer full results](reference/results.json) from the sealed server release. GCN and GAT completed through the reviewer-account workflow on 2026-09-24: each arm ran 20 epochs, 20 full validations and one final test; all workers exited normally and strict monitors had zero query errors. The [full acceptance receipt](reference/reviewer_full.json) binds 202 independently checked evidence hashes. DiGiT test accuracy is lower by 0.3788 percentage points for GCN and 0.6630 for GAT. One seed does not establish statistical accuracy equivalence or the paper's absolute accuracy. The [previous index](reference/results_before_reviewer_full_20260924.json) preserves original GCN/GAT results, including four GCN GIDS monitor timeouts. [Earlier reviewer smoke](reference/reviewer_smoke.json) and rebuilt-source acceptance remain separate. See [metric definitions](docs/RESULTS.md).

### PA/SAGE component ablation

Accepted AE self-service measurements, completed 2026-09-25: one full seed-0 training epoch per arm, without validation/test.

| Stage | Training time (s) | Speedup vs GIDS |
|---|---:|---:|
| GIDS | 194.63 | 1.0000× |
| +GR (adjacency only) | 196.12 | 0.9924× |
| ++NS | 160.38 | 1.2136× |
| DiGiT | 103.69 | 1.8770× |

Four-arm smoke and full acceptance passed with normal exits and zero monitor query errors. All arms use a 4 GiB GPU feature cache; the first three share the main RevPR CPU hot set. +GR changes adjacency order only; GR→NS also changes feature layout. Training excludes preparation and setup.

[Full results and protocol](docs/ABLATION.md) · [Download CSV](reference/pa_sage_ablation_perf.csv) · [AE acceptance evidence](reference/pa_sage_ablation_perf.json) · [Earlier author results](reference/pa_sage_ablation_author_20260924.json)

The [validation receipt](provenance/validation.json) records 22 model/budget checks, eight entry/counter boundary checks, a three-model CPU example, exact environment checks, and source/configuration parity checks. A subsequent [fresh-install and native-build receipt](provenance/native_build_validation.json) records successful locked environment installation, the three-model CPU example, compilation and module imports. After restoring the omitted `GIDS.breakdown` dependency, GPU/SSD preflight and paired smoke passed for all three models: six workers exited normally, strict monitors reported zero query errors, and 130 evidence hashes matched. Each arm performed four training updates and two limited validation calls, with no final test. The [rebuilt-source receipt](reference/rebuilt_native_smoke.json) records the accepted package identity and binary reuse; this does not claim new full experiments or reviewer-account workflow acceptance. Earlier validation receipts retain their original scope.

### PA/SAGE grouping and replication

Accepted author measurements: one complete first epoch per point, using shared proxy
features and real sampling/I/O/model computation. Training seconds:

| Group size | 0% | 10% | 20% | 40% | 80% |
|---|---:|---:|---:|---:|---:|
| g = 1 | 120.09 | 119.79 | 120.61 | 122.11 | 122.33 |
| g = 2 | 105.27 | 104.18 | 104.34 | 104.56 | 104.49 |
| g = 4 | 92.58 | 91.18 | 92.58 | 92.09 | 90.21 |

[Protocol, verification scope and AE replay commands](docs/LAYOUT_GRID.md) ·
[JSON](reference/pa_sage_layout_grid.json) · [CSV](reference/pa_sage_layout_grid.csv).
The author grid is accepted; a fresh AE extension replay remains separately verifiable.

## Set up from source

Start with Linux x86_64, Git, Python 3 and Conda. The [environment guide](docs/ENVIRONMENT.md) lists the recorded software versions, installation prerequisites and expected check results. Use a new environment directory outside the source checkout:

```bash
git clone https://github.com/starry12/DiGiT.git
cd DiGiT
bash run.sh verify
bash run.sh matrix

bash environment/create.sh "$HOME/digit-env"
export DIGIT_PYTHON="$HOME/digit-env/bin/python"
bash run.sh environment
bash run.sh example --output results/cpu_example_01
```

The CPU example runs three updates with each selected model and optimizer. It needs no dataset, GPU or raw SSD. Use a new output directory for each invocation. Next, follow the [native build guide](docs/NATIVE_BUILD.md) to compile the CUDA and storage components, then the [data guide](docs/DATA.md) to bind prepared inputs before native preflight and paired smoke. The environment installer covers Python dependencies; CUDA, the NVIDIA driver and the libnvm kernel module are separate host prerequisites.

## Source layout

- `training/sage`, `training/gcn`, `training/gat`: selected protocols, models, native workers and validation.
- `evaluation/`: orchestration, acceptance and summaries; the single user entry is `run.sh`.
- `runtime/io`: useful I/O counters and the instrumented feature store.
- `ae/`: shared graph, evaluation, monitoring and model support.
- `third_party/bam`: one vendored BaM dependency, including original notices.
- `environment/`, `configs/`, `scripts/`: dependency locks, data contracts and build/binding helpers.
- `reference/`: selected result summaries and the necessary deterministic SAGE correctness oracle.

[Data](docs/DATA.md) and [native build instructions](docs/NATIVE_BUILD.md) describe the prepared-input contract. A fresh end-to-end dataset download/preparation pipeline and a container deployment have not been validated. The current prepared server supports the PA main experiments and the supplementary PA/SAGE ablation and layout grid above. IG, the web graphs and the remaining sensitivity/scalability experiments are outside this evaluation. The submitted `ae-pa-v1` remains the frozen initial PA snapshot; these supplementary updates are on `main`. Project licensing is recorded in [LICENSE_STATUS.md](LICENSE_STATUS.md).
