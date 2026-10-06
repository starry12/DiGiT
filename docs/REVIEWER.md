# Reviewer workflow

Activate the prepared environment. No compilation, data preparation or administrator password is needed.

```bash
source /srv/digit-ae/activate.sh
```

| Experiment | Start | Inspect results |
|---|---|---|
| PA / GraphSAGE | `digit-ae run PA sage` | `digit-ae results PA sage --action run` |
| PA / GCN | `digit-ae run PA gcn` | `digit-ae results PA gcn --action run` |
| PA / GAT | `digit-ae run PA gat` | `digit-ae results PA gat --action run` |
| PA / GraphSAGE ablation | `digit-ae ablation PA sage` | `digit-ae results PA sage --action ablation` |
| PA / GraphSAGE layout grid | `digit-ae layout PA sage` | `digit-ae results PA sage --action layout` |
| IG / GraphSAGE | `digit-ae performance IG sage` | `digit-ae results IG sage --action performance` |
| UKS / GraphSAGE | `digit-ae performance UKS sage` | `digit-ae results UKS sage --action performance` |
| UKL / GraphSAGE | `digit-ae performance UKL sage` | `digit-ae results UKL sage --action performance` |
| CL / GraphSAGE | `digit-ae performance CL sage` | `digit-ae results CL sage --action performance` |

Run one request at a time. Each request selects an idle GPU from cards 0–3 and retains it throughout the experiment. GPU and SSD locks prevent concurrent AE requests. Busy requests are rejected; they are not queued. An accepted launch continues after SSH disconnects.

Replace `results` by `status` or `logs` to inspect progress. To cancel, replace it by `stop`, keeping the same dataset, model and `--action`. Wait for the service to become inactive before starting another request.

`PASS` requires accepted reports and successful service completion. Default commands inspect the latest request, including a failed or incomplete one. Add `--reference` to the layout, IG or UKS **results** command to read the selected accepted reference without launching anything. Reference output is labeled `AE_REFERENCE`; it is not a fresh execution.

The output directory printed by the CLI contains detailed evidence. Use `--json` with status/results for machine-readable records, or download files through the supplied SSH/SFTP connection. Default result display uses speedups; PA main results also show test accuracy.

An optional short check is `digit-ae smoke PA sage` (also `gcn` or `gat`). Full PA requests already include smoke, then 20 epochs per system, validation each epoch and one final test. Ablation and grid use one full epoch per setting. IG, UKS, UKL and CL use five pairs of 20 warmup and 300 measured batches.

See the [homepage](../README.md) for result tables, [scope](CLAIMS.md) for measurement boundaries, and [code map](CODE.md) for the implementation. Resource or device errors should be reported with the printed output path to the server administrator; reviewers need not repair the environment.
