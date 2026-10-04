# PA/SAGE grouping and replication grid

```bash
digit-ae layout PA sage
digit-ae status PA sage --action layout
digit-ae results PA sage --action layout
digit-ae logs PA sage --action layout
```

The service reuses 15 prepared read-only layouts and a verified shared proxy-feature SSD region. It runs fresh workers for group sizes {1, 2, 4} × replication ratios {0%, 10%, 20%, 40%, 80%}. Every point completes one seed-0 epoch, 1,207,179 roots and 1,179 updates, without validation/test. One idle GPU from cards 0–3 is retained for the whole request.

| Group size | 0% | 10% | 20% | 40% | 80% |
|---|---:|---:|---:|---:|---:|
| g = 1 | 0.88× | 0.88× | 0.88× | 0.87× | 0.86× |
| g = 2 | 0.99× | 1.00× | 1.00× | 1.01× | 1.00× |
| g = 4 | 1.14× | 1.16× | 1.15× | 1.15× | 1.16× |

Speedups use g2/r20 in the same accepted run as baseline. All 15 points come from one complete run. The fastest point at full precision is g4/r80; this does not establish a statistical optimum.

Sampling, physical I/O and forward/backward/Adam computation are real. Features use shared physical-row proxy data. A real/shared g2/r20 calibration passed; this does not establish logical-feature equivalence for every layout or support accuracy claims.

g2/r20 retains semantic checks and a fresh native smoke. The other 14 points retain address bounds, full epoch coverage, finite values, I/O accounting, monitoring and normal-exit checks. Graph construction and fresh preprocessing time are outside this replay.

Use `digit-ae results PA sage --action layout --reference` to read the accepted `AE_REFERENCE` without starting a request. Default results inspect the latest request, including failure.

[Accepted result](../reference/pa_sage_layout_grid.json) · [Acceptance receipt](../reference/pa_sage_layout_reviewer_acceptance.json) · [Source](../tools/layout_grid/README.md).
