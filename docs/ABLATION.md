# PA/SAGE supplementary component ablation

Measured on 2026-09-24 with an independent author candidate. Four native smoke workers and four fresh performance workers passed; all exited with code 0, with zero monitor query errors. Each performance arm completed one entire training epoch (1,207,179 examples; 1,179 updates), without validation/test.

| Stage | CPU cache rows | Training (s) | Speedup vs this run GIDS |
|---|---:|---:|---:|
| GIDS (no CPU cache) | 0 | 227.37 | 1.0000x |
| +GR | 0 | 167.93 | 1.3539x |
| ++NS | 0 | 181.51 | 1.2527x |
| DiGiT | 11105992 | 105.60 | 2.1531x |

The cumulative stages are original layout/standard sampling → grouped layout with standard sampling (+GR) → grouped sampling and mixed I/O (++NS) → static CPU hot cache plus FIFO GPU replacement (DiGiT). The final step measures the cache package, not FIFO alone. GPU cache is 4 GiB for every arm; BFS stays disabled.

**NS is 8.09% slower than GR.** This negative result is retained. The 2.1531x DiGiT speedup uses the actual 227.37 s GIDS measurement with CPU cache disabled. It does not use the historical 195.48 s GIDS mean. The main PA/SAGE result remains 1.8027x: its GIDS includes static CPU cache, and its times cover 20 training epochs. Both ratios exclude validation/test time.

Timing is first-epoch training wall time, including root-order generation, sampling, feature fetching, forward/backward and optimizer updates. Preparation, startup, metadata setup, smoke and process teardown are separate. Fresh processes have no explicit training warmup; CPU preload follows each policy. No claim is made about cache steady state, repeat variance or accuracy.

The [curated receipt](../reference/pa_sage_ablation_perf.json) records controls, measured rows, I/O, setup costs, worker acceptance and evidence hashes. The [candidate manifest](../provenance/pa_sage_ablation_perf_manifest.json) identifies the author source and launcher used. These are accepted author results, not new measurements of this refactored checkout. Its four-arm controller is not integrated into `run.sh` or the sealed reviewer CLI. The submitted AE main matrix and frozen tag remain separate.
