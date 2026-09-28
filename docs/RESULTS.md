# Reading results

| Experiment | Final speedup |
|---|---:|
| PA / GraphSAGE main comparison | 1.80× |
| PA / GCN main comparison | 1.86× |
| PA / GAT main comparison | 1.73× |
| PA / GraphSAGE component ablation, final DiGiT vs GIDS | 1.88× |
| PA / GraphSAGE best layout vs g2/r20 | 1.16× |
| IG / GraphSAGE, maximum paired speedup across five rounds | 1.46× |

PA main results are accepted reviewer comparisons over 20 training epochs.
The accepted PA ablation uses one full epoch per arm. The layout result selects
g4/r80 from the accepted author grid, relative to g2/r20, rather than GIDS.
IG is an accepted AE reviewer short-window result: five paired rounds, each with
20 warmup and 300 timed mini-batches. Its maximum observed speedup does not
establish average or stable performance. The IG AE replay has passed; fresh AE layout replay remains pending.

[Metric definitions](RESULTS.md) · [Ablation protocol](ABLATION.md) ·
[Layout protocol](LAYOUT_GRID.md) · [IG protocol](IG_PERFORMANCE.md).
Default presentation shows final speedups; machine-readable audit evidence is retained.

Main speedup divides GIDS total training time by DiGiT total training time over
20 epochs. Ablation uses the measured GIDS first epoch as its denominator; layout
uses g2/r20. IG uses the maximum same-round ratio across five 300-batch runs.
Preparation, validation and test are outside training speedup. These scopes are
not interchangeable. No multi-seed accuracy equivalence is claimed.

Existing reference JSON/CSV and acceptance receipts retain the complete audit
record, including historical failures. Default human-readable summaries omit
per-run timing details; PA main test accuracy remains in the README. Use explicit JSON/evidence inspection when needed.
