# PA/SAGE grouping and replication grid

The selected **AE reviewer** grid completed on 2026-09-29 at 09:57:49 UTC+8.
All 15 points passed: one complete seed-0 epoch, 1,207,179 roots and 1,179 updates
per point, without validation/test. This is the entire second AE run selected at
the author's request; the earlier run is preserved, without mixing points across runs.

Speedups use this run's **g2/r20** as baseline:

| Group size | 0% | 10% | 20% | 40% | 80% |
|---|---:|---:|---:|---:|---:|
| g = 1 | 0.88× | 0.88× | 0.88× | 0.87× | 0.86× |
| g = 2 | 0.99× | 1.00× | 1.00× | 1.01× | 1.00× |
| g = 4 | 1.14× | 1.16× | 1.15× | 1.15× | 1.16× |

The fastest point is g4/r80 (**1.16×** at full precision before rounding).
No statistical optimum or accuracy claim is made.

## Feature and verification scope

All points retain real graph sampling, physical I/O and forward/backward/Adam computation,
using a shared physical-row proxy feature region. A real/shared g2/r20 ABBA calibration
passed after explicitly correcting a service-path-counter equality check: logical
requests must match, while CPU/GPU service splits can vary with cache residency.
The original time/I/O gates remained unchanged; the original failed gate is preserved.
This calibration does not prove every new layout equivalent to real logical features.

g2/r20 retains complete semantic checks and independent native smoke. The other 14
points skip those extra passes by author decision, while retaining address bounds,
full epoch coverage, finite training values, I/O accounting, monitoring and normal exit.
The reported layout address span is not newly materialized feature-file size.

## Prepared-server replay

The installed and accepted extension reuses the
15 prepared read-only layouts and verified SSD pool; it never rebuilds graphs or writes
raw SSD data. A request creates a new output directory and fresh workers for all 15
full epochs, with a new g2/r20 smoke. It does not reuse author performance reports.
Source publication, installation/import checks and a successful AE replay are separate.

```bash
digit-ae layout PA sage
digit-ae status PA sage --action layout
digit-ae results PA sage --action layout
digit-ae logs PA sage --action layout
digit-ae stop PA sage --action layout
```

To inspect the selected accepted AE measurements explicitly:

```bash
digit-ae results PA sage --action layout --reference
```

Reference output is labelled AE_REFERENCE, never fresh AE PASS. Without --reference,
the command reads only the latest AE request, including failure. PASS requires all 15
accepted reports, zero monitor errors and successful service exit. GPU 2 and data paths
are fixed. The whole request owns the existing AE/NVMe exclusion locks. Services survive
SSH disconnects; other experiments must wait. Both complete AE runs are preserved; the second is the current published reference.

The original nvidia-smi monitor retains a 5-second individual-query timeout and 0.5-second
wait after queries; no 5.5-second inter-sample rejection is added. Preparation costs remain
separate. This prepared-layout replay does not measure fresh preprocessing time.

## Source and evidence

- [Receipt](../reference/pa_sage_layout_grid.json) and [full-precision CSV](../reference/pa_sage_layout_grid.csv).
- [Source/evidence index](../provenance/pa_sage_layout_grid_manifest.json).
- [AE extension and exact supplementary source](../tools/layout_grid/README.md).

The main PA 20-epoch results, accepted ablation and submitted ae-pa-v1 remain unchanged.
The seven independent CPU-only preparation candidates are not included in this update.

[AE acceptance receipt](../reference/pa_sage_layout_reviewer_acceptance.json).
