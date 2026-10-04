# PA/SAGE component ablation

```bash
digit-ae ablation PA sage
digit-ae status PA sage --action ablation
digit-ae results PA sage --action ablation
digit-ae logs PA sage --action ablation
```

Each request runs four fresh smoke workers, then GIDS → +GR → ++NS → DiGiT for one complete seed-0 training epoch each: 1,207,179 examples and 1,179 updates, without validation/test. An idle GPU is selected from cards 0–3 for the whole request.

| Arm | Speedup vs GIDS |
|---|---:|
| GIDS | 1.00× |
| +GR | 0.99× |
| ++NS | 1.21× |
| DiGiT | 1.88× |

All arms use a 4 GiB GPU feature cache and 11,105,992 CPU-cached rows. The first three share the main GIDS RevPR logical hot set. +GR changes adjacency order only, retaining node IDs, original feature storage and cache lookup. ++NS adds grouped storage and sampling with row-exact mapping of that hot set. GR→NS therefore also changes feature layout. DiGiT changes CPU hot-set selection and GPU replacement together.

Timing includes root-order generation, sampling, feature fetch and model updates. Preparation, initialization and teardown are separate. The single-epoch result makes no steady-state or accuracy claim. All eight workers in the accepted reference exited successfully.

[Accepted result](../reference/pa_sage_ablation_perf.json) · [Source](../tools/ablation/README.md) · [Reviewer workflow](REVIEWER.md).
