# Results

## PA main experiments

| Model | Speedup vs GIDS | GIDS test accuracy | DiGiT test accuracy |
|---|---:|---:|---:|
| GraphSAGE | 1.80× | 62.85% | 62.96% |
| GCN | 1.86× | 53.74% | 53.36% |
| GAT | 1.73× | 48.62% | 47.96% |

Speedup divides GIDS total training time by DiGiT total training time over 20 epochs. Preparation, validation and final test are outside that timing. Accuracy is measured using each epoch-20 checkpoint, seed 0.

## PA supplementary performance

| Experiment | Speedup | Baseline |
|---|---:|---|
| Four-arm ablation, final DiGiT | 1.88× | Measured GIDS arm |
| Best grouping/replication point | 1.16× | g2/r20 in the same accepted grid |

Ablation and layout use one full epoch per point. The layout table comes from one complete accepted run; points are not mixed across runs.

## Short-window performance

| Experiment | Maximum paired speedup vs GIDS |
|---|---:|
| IG / GraphSAGE | 1.46× |
| UKS / GraphSAGE | 2.00× |
| UKL / GraphSAGE | 1.70× |
| CL / GraphSAGE | 1.65× |

Each comparison uses five paired rounds, 20 warmup and 300 timed mini-batches per system. The reported statistic is the maximum same-round GIDS/DiGiT ratio, computed before rounding; it is not mean performance. UKS, UKL and CL use synthetic features and labels and make no accuracy claim. UKL/CL are validated author results; reviewer-account replay is pending.

Full tables and commands are on the [homepage](../README.md). Current acceptance records are retained in `reference/`; raw timing and I/O records remain available in each server result directory. Ratios are displayed to two decimal places.

[PA acceptance](../reference/reviewer_full.json) · [Ablation](ABLATION.md) · [Layout](LAYOUT_GRID.md) · [IG](IG_PERFORMANCE.md) · [UKS](UKS_PERFORMANCE.md).
