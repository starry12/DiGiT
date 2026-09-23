# DiGiT: Papers100M artifact evaluation

This repository contains the selected **Papers100M (PA)** implementation for **GraphSAGE, GCN and GAT**, comparing **GIDS and DiGiT**. Each full comparison uses seed 0, 20 epochs, full validation and one final test. We report training time, accuracy and effective I/O. This is a reconstruction of the paper implementation; BFS is disabled.

The source tree contains one selected implementation per model. It excludes research Git history, intermediate implementations, training logs, checkpoints, compiled binaries and datasets. See [scope and claims](docs/CLAIMS.md) and [code organization](docs/CODE.md).

## Quick check

```bash
bash run.sh verify
bash run.sh matrix
export DIGIT_PYTHON=/srv/digit-ae/env/bin/python  # on the provided server
bash run.sh environment
bash run.sh example --output results/cpu_example_01
```

The CPU example runs three updates with each selected model and optimizer. It needs no dataset, GPU or raw SSD. Use a new output directory for each invocation. On another machine, follow [environment setup](docs/ENVIRONMENT.md).

## Reproduce on the provided server

SSH access is provided privately through the AE channel. After login:

```bash
source /srv/digit-ae/activate.sh
digit-ae smoke PA sage
digit-ae status PA sage
digit-ae results PA sage --action smoke
```

After smoke passes, start a full comparison with `digit-ae run PA sage` and read it with `digit-ae results PA sage --action run`. Substitute `gcn` or `gat` for the other models; run requests sequentially. Services survive SSH disconnects. [Reviewer instructions](docs/REVIEWER.md) explain progress, completion and result export.

The service currently executes the preserved, accepted server release. This repository reorganizes that release into a concise source tree; its native rebuild and native acceptance are tracked separately. Source origins and adaptations are recorded in [the source map](provenance/source_map.json). The service is not silently upgraded when this repository changes.

## Results and boundaries

| Model | Selected training speedup | GIDS test accuracy | DiGiT test accuracy | Evidence |
|---|---:|---:|---:|---|
| GraphSAGE | 1.8027× | 62.8531% | 62.9552% | Fresh reviewer full pair; strict monitoring passed |
| GCN | 1.8868× | 53.7422% | 53.3634% | Author full pair; GIDS had four monitor query timeouts |
| GAT | 1.7397× | 48.6204% | 47.9574% | Author full pair; strict monitoring passed |

These selected measurements are [reference evidence](reference/results.json), not measurements of newly rebuilt binaries. Fresh reviewer GCN/GAT preflight and paired smoke have also passed, with error-free monitoring and normal service exit; see [smoke acceptance](reference/reviewer_smoke.json). These short checks do not replace full performance/accuracy evidence. One seed does not establish statistical accuracy equivalence or the paper's absolute accuracy. See [metric definitions](docs/RESULTS.md).

The [validation receipt](provenance/validation.json) records 22 model/budget checks, eight entry/counter boundary checks, a three-model CPU example, exact environment checks, and source/configuration parity checks. Native compilation and execution of this refactored source are still pending.

## Source layout

- `training/sage`, `training/gcn`, `training/gat`: selected protocols, models, native workers and validation.
- `evaluation/`: orchestration, acceptance and summaries; the single user entry is `run.sh`.
- `runtime/io`: useful I/O counters and the instrumented feature store.
- `ae/`: shared graph, evaluation, monitoring and model support.
- `third_party/bam`: one vendored BaM dependency, including original notices.
- `environment/`, `configs/`, `scripts/`: dependency locks, data contracts and build/binding helpers.
- `reference/`: selected result summaries and the necessary deterministic SAGE correctness oracle.

[Data](docs/DATA.md) and [native build instructions](docs/NATIVE_BUILD.md) describe the prepared-input contract. A fresh end-to-end dataset download/preparation pipeline and a container deployment have not been validated. IG and the web graphs, ablations, sensitivity and scalability are outside this submission. Project licensing is recorded in [LICENSE_STATUS.md](LICENSE_STATUS.md).
