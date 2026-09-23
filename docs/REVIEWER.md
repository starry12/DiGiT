# Reviewer workflow

Use the provided server for native GPU/NVMe experiments. Connection details are supplied privately. The commands do not require an administrator password.

```bash
source /srv/digit-ae/activate.sh
digit-ae smoke PA sage
```

The service first validates the model/data/resource contract and then runs the short GIDS/DiGiT pair. It uses fixed GPU 2 and an exclusive GPU/SSD lock. A busy device is rejected; requests are not queued by the service. Do not start another model until the current one finishes.

```bash
digit-ae status PA sage
digit-ae logs PA sage
digit-ae results PA sage --action smoke
```

A `RUNNING` table is provisional. `PASS`, paired acceptance, strict resource monitoring and a normally exited service are required for acceptance. Earlier success is not substituted for an unfinished/failed run. The output path printed by the CLI contains the JSON reports and counters. Use `--json` with `results` for machine-readable output.

For the complete experiment:

```bash
digit-ae run PA sage
digit-ae results PA sage --action run
```

This repeats paired smoke before two fresh 20-epoch workers; each worker validates after each epoch and tests its final model once. GraphSAGE's accepted workflow took about 2 h 25 min on this server; runtime varies. GCN/GAT can take longer. Services survive disconnected terminals. To explicitly stop a run, use `digit-ae stop PA sage`.

Replace `sage` with `gcn` or `gat` for those models. The serial GCN-then-GAT reviewer smoke queue completed successfully. Both preflights and both system arms passed strict monitoring with zero query timeouts; see [smoke acceptance](../reference/reviewer_smoke.json). Each smoke arm made four training updates, two bounded validation calls and no final test. These are usability/correctness checks, not new full performance or accuracy measurements.

The server release remains separately sealed. `activate.sh` changes into that server directory. To inspect this concise repository, return to your clone and consult [CODE.md](CODE.md). Do not equate a newly built checkout with an already accepted service binary; each new deployment needs its own preflight/smoke evidence.
