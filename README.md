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

**Run one request at a time:** main models, ablation, the layout grid and IG share GPU 2 and the SSD. Wait for the current request to finish and release resources before starting the next; busy requests are rejected rather than queued. Each launch creates fresh results and continues after SSH disconnects. No local build or data preparation is needed on this server.

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

After installation of the concise result display, accepted results show the final DiGiT/GIDS speedup. The existing AE run has passed; [protocol and evidence](docs/ABLATION.md) describe the cache settings and component boundaries.

### Grouping and replication grid: PA / GraphSAGE

This request covers **group sizes {1, 2, 4} × replication ratios {0%, 10%, 20%, 40%, 80%}**, for **15 fresh full-epoch measurements (1,179 updates each), without validation/test**:

```bash
digit-ae layout PA sage
digit-ae status PA sage --action layout
digit-ae results PA sage --action layout
digit-ae logs PA sage --action layout
```

The service reuses the 15 prepared layouts and shared proxy-feature SSD region, with real sampling, I/O and model updates. It runs a fresh g2/r20 smoke before the full grid; the other 14 points retain the declared runtime checks. It does not regenerate large layouts. Status reports completion out of 15; accepted results show the best layout speedup relative to g2/r20.

The published author grid is accepted and the AE extension is installed; a fresh AE grid replay has not yet been accepted. To view the author measurements without starting a run:

```bash
digit-ae results PA sage --action layout --reference
```

This explicitly prints `AUTHOR_REFERENCE`, separate from a new AE request's `PASS`. See [grid protocol and evidence](docs/LAYOUT_GRID.md) for proxy-feature calibration and verification scope.

### IG / GraphSAGE performance: five paired rounds

The IG extension adds a performance-only comparison: **five paired rounds**, each
with **20 warmup and 300 timed mini-batches** per system. The extension is installed and the AE five-round replay has passed:


```bash
digit-ae performance IG sage
digit-ae status IG sage --action performance
digit-ae results IG sage --action performance
digit-ae logs IG sage --action performance
```

Results show the **maximum same-round GIDS/DiGiT speedup across five rounds**.
This is a short-window performance measurement, not an accuracy or full-epoch run.
To inspect the accepted AE reference without running a request:
`digit-ae results IG sage --action performance --reference`.
The reference is labeled `AE_REFERENCE`; default results inspect the latest request.
[IG protocol and source](docs/IG_PERFORMANCE.md).

### Completion, stopping and result files

`status`, `logs` and `results` only inspect records. Use the matching `--action run`, `--action ablation` `--action layout` or `--action performance` above to select the experiment. They select the latest request for that action, including a failed one. Final **`PASS` requires accepted reports and successful service completion**; a launch acknowledgement or running/provisional table is not final acceptance. The printed output path contains the logs, reports and summaries and can be downloaded with SFTP/SCP through the supplied SSH route.

To cancel a request, use its matching command and wait for inactive status and resource release:

| Request | Stop command |
|---|---|
| Main comparison | `digit-ae stop PA sage` (substitute `gcn` or `gat`) |
| Four-arm ablation | `digit-ae stop PA sage --action ablation` |
| Layout grid | `digit-ae stop PA sage --action layout` |
| IG performance | `digit-ae stop IG sage --action performance` |

The main SAGE full workflow previously took about **2 h 25 min**, and four-arm ablation about **59 min**, including smoke and setup. These are observed wall times, not estimates from the per-epoch result tables; server load can change them. [Reviewer instructions](docs/REVIEWER.md) provide additional troubleshooting and environment details.

The main experiments execute the preserved, accepted server release; the supplementary services use separately installed runtime snapshots. This repository reorganizes the main release into a concise source tree; its native compilation in a fresh locked environment and subsequent PA/SAGE, GCN and GAT preflight and paired smoke have passed; see the [rebuilt-source receipt](reference/rebuilt_native_smoke.json). Source origins and adaptations are recorded in [the source map](provenance/source_map.json). Server runtimes are updated through explicit installation.

## Current evaluation scope

The current artifact evaluates **Papers100M (PA)** with GraphSAGE, GCN and GAT, comparing GIDS and DiGiT. Each main comparison uses seed 0, 20 epochs, full validation and one final test. The supplementary PA/SAGE ablation and grouping/replication grid use one complete epoch per setting and report performance only. The supplementary IG/SAGE comparison uses five paired short performance windows. This is a reconstruction of the paper implementation.

The source tree contains one selected implementation per model. It excludes research Git history, intermediate implementations, training logs, checkpoints, compiled binaries and datasets.

## Results and boundaries

### PA main experiments

Accepted reviewer comparisons over 20 training epochs per system. Speedup is GIDS training time divided by DiGiT training time.

| Model | Training speedup vs GIDS | GIDS test accuracy | DiGiT test accuracy |
|---|---:|---:|---:|
| GraphSAGE | 1.80× | 62.85% | 62.96% |
| GCN | 1.86× | 53.74% | 53.36% |
| GAT | 1.73× | 48.62% | 47.96% |

Test accuracy is measured once using each system’s epoch-20 checkpoint (seed 0). These single-seed results do not establish statistical accuracy equivalence.

[Main results and metric definitions](docs/RESULTS.md).

### PA/SAGE component ablation

Accepted AE measurements: one complete training epoch per arm, without validation/test. All speedups use the measured GIDS arm as the baseline.

| Stage | Speedup vs GIDS |
|---|---:|
| GIDS | 1.00× |
| +GR (adjacency only) | 0.99× |
| ++NS | 1.21× |
| DiGiT | 1.88× |

+GR changes adjacency order only; GR→NS also changes feature layout.
[Ablation protocol and evidence](docs/ABLATION.md).

### PA/SAGE grouping and replication

Accepted author grid: one complete first epoch per point, with shared proxy features and real sampling, I/O and model computation. Speedups are relative to **g2/r20**, not GIDS.

| Group size | 0% | 10% | 20% | 40% | 80% |
|---|---:|---:|---:|---:|---:|
| g = 1 | 0.87× | 0.87× | 0.87× | 0.85× | 0.85× |
| g = 2 | 0.99× | 1.00× | 1.00× | 1.00× | 1.00× |
| g = 4 | 1.13× | 1.14× | 1.13× | 1.13× | 1.16× |

[Grid protocol and evidence](docs/LAYOUT_GRID.md). Fresh AE grid replay acceptance remains separate from the author results.

### IG/SAGE performance comparison

Accepted AE reviewer short-window comparison: five paired rounds, each with 20 warmup and 300 timed mini-batches per system.

| Model | Maximum paired speedup vs GIDS (five rounds) |
|---|---:|
| GraphSAGE | 1.46× |

The reported value is the maximum same-round GIDS/DiGiT ratio, not average or stable performance. GIDS uses default CPU scheduling; DiGiT binds to CPU2 after imports. No accuracy or full-epoch claim is made. All ten workers passed; the service completed successfully on 2026-09-28.

[IG protocol and evidence](docs/IG_PERFORMANCE.md) · [AE acceptance receipt](reference/ig_sage_reviewer_acceptance.json).

All displayed ratios are rounded to two decimal places; full-precision audit evidence is retained.

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

[Data](docs/DATA.md) and [native build instructions](docs/NATIVE_BUILD.md) describe the prepared-input contract. A fresh end-to-end dataset download/preparation pipeline and a container deployment have not been validated. The current prepared server supports the PA main experiments and the supplementary PA/SAGE ablation and layout grid above. IG/SAGE has the supplementary short-window performance protocol above; web graphs and the remaining sensitivity/scalability experiments are outside this evaluation. The submitted `ae-pa-v1` remains the frozen initial PA snapshot; these supplementary updates are on `main`. Project licensing is recorded in [LICENSE_STATUS.md](LICENSE_STATUS.md).
