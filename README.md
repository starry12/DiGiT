# DiGiT

DiGiT is an out-of-core graph neural network training system built on GIDS. It combines GPU-side neighbor sampling, SSD-backed feature access and multilevel caching to train graphs whose features exceed GPU memory.

This repository provides the DiGiT implementation, the GIDS comparison path, and experiment entry points for **GraphSAGE, GCN and GAT**. Measurements cover training time, accuracy and effective I/O. See the [code map](docs/CODE.md) for the implementation and [scope and claims](docs/CLAIMS.md) for the current evaluation coverage.

## Getting started

| What you want to do | Start here |
|---|---|
| Run experiments on the provided AE server | [Reviewer guide](docs/REVIEWER.md) and [AE server commands](#ae-environment-and-reproduction) |
| Create an environment from scratch on your own machine | [Environment setup](docs/ENVIRONMENT.md): prerequisites, locked dependencies and checks |
| Compile DiGiT and its GIDS/BaM dependencies | [Native build guide](docs/NATIVE_BUILD.md) |
| Connect the prepared graph, features and SSD data | [Data guide](docs/DATA.md) |

## AE environment and reproduction

The provided AE server includes the Python/CUDA environment, compiled native components and prepared data. SSH access is provided privately through the AE channel. After login, activate the prepared environment and start the short comparison:

```bash
source /srv/digit-ae/activate.sh
digit-ae smoke PA sage
digit-ae status PA sage
digit-ae results PA sage --action smoke
```

After smoke passes, start a full comparison with `digit-ae run PA sage` and read it with `digit-ae results PA sage --action run`. Substitute `gcn` or `gat` for the other models; run requests sequentially. Services survive SSH disconnects. [Reviewer instructions](docs/REVIEWER.md) explain progress, completion and result export.

The service currently executes the preserved, accepted server release. This repository reorganizes that release into a concise source tree; its native compilation in a fresh locked environment and subsequent PA/SAGE, GCN and GAT preflight and paired smoke have passed; see the [rebuilt-source receipt](reference/rebuilt_native_smoke.json). Source origins and adaptations are recorded in [the source map](provenance/source_map.json). The service is not silently upgraded when this repository changes.

## Current evaluation scope

The current artifact evaluates **Papers100M (PA)** with GraphSAGE, GCN and GAT, comparing GIDS and DiGiT. Each full comparison uses seed 0, 20 epochs, full validation and one final test. This is a reconstruction of the paper implementation; BFS is disabled.

The source tree contains one selected implementation per model. It excludes research Git history, intermediate implementations, training logs, checkpoints, compiled binaries and datasets.

## Results and boundaries

| Model | Selected training speedup | GIDS test accuracy | DiGiT test accuracy | Evidence |
|---|---:|---:|---:|---|
| GraphSAGE | 1.8027× | 62.8531% | 62.9552% | Fresh reviewer full pair; strict monitoring passed |
| GCN | 1.8584× | 53.7422% | 53.3634% | Fresh reviewer full pair; strict monitoring passed |
| GAT | 1.7266× | 48.6204% | 47.9574% | Fresh reviewer full pair; strict monitoring passed |

These selected measurements are [fresh reviewer full results](reference/results.json) from the sealed server release. GCN and GAT completed through the reviewer-account workflow on 2026-09-24: each arm ran 20 epochs, 20 full validations and one final test; all workers exited normally and strict monitors had zero query errors. The [full acceptance receipt](reference/reviewer_full.json) binds 202 independently checked evidence hashes. DiGiT test accuracy is lower by 0.3788 percentage points for GCN and 0.6630 for GAT. One seed does not establish statistical accuracy equivalence or the paper's absolute accuracy. The [previous index](reference/results_before_reviewer_full_20260924.json) preserves original GCN/GAT results, including four GCN GIDS monitor timeouts. [Earlier reviewer smoke](reference/reviewer_smoke.json) and rebuilt-source acceptance remain separate. See [metric definitions](docs/RESULTS.md).

The [validation receipt](provenance/validation.json) records 22 model/budget checks, eight entry/counter boundary checks, a three-model CPU example, exact environment checks, and source/configuration parity checks. A subsequent [fresh-install and native-build receipt](provenance/native_build_validation.json) records successful locked environment installation, the three-model CPU example, compilation and module imports. After restoring the omitted `GIDS.breakdown` dependency, GPU/SSD preflight and paired smoke passed for all three models: six workers exited normally, strict monitors reported zero query errors, and 130 evidence hashes matched. Each arm performed four training updates and two limited validation calls, with no final test. The [rebuilt-source receipt](reference/rebuilt_native_smoke.json) records the accepted package identity and binary reuse; this does not claim new full experiments or reviewer-account workflow acceptance. Earlier validation receipts retain their original scope.

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

[Data](docs/DATA.md) and [native build instructions](docs/NATIVE_BUILD.md) describe the prepared-input contract. A fresh end-to-end dataset download/preparation pipeline and a container deployment have not been validated. IG and the web graphs, ablations, sensitivity and scalability are outside this submission. Project licensing is recorded in [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Supplementary author experiments

[PA/SAGE component ablation](docs/ABLATION.md) now has accepted one-epoch performance results: GIDS (CPU cache disabled), +GR, ++NS and DiGiT took **227.37 / 167.93 / 181.51 / 105.60 s**. DiGiT is **2.1531x** faster than this measured GIDS baseline; NS is **8.09% slower** than GR. This supplements the main results and does not replace their CPU-cache-enabled GIDS reference or accuracy protocol. [Protocol and acceptance receipt](reference/pa_sage_ablation_perf.json). The four-arm author controller is not integrated into the published or reviewer CLI.
