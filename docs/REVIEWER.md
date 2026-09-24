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

Replace `sage` with `gcn` or `gat` for those models. All three models now have accepted full reviewer pairs. GCN followed by GAT completed on 2026-09-24 in 5 h 10 min 20 s combined, including repeated smoke and initialization. Each full arm completed 20 epochs, 20 full validations and a final test, with zero monitor query errors; see [full acceptance](../reference/reviewer_full.json). Use `digit-ae results PA gcn --action run` or the corresponding GAT command to inspect existing results without launching another run. The CLI selects the latest request; the receipt identifies the fixed accepted directories. [Earlier smoke acceptance](../reference/reviewer_smoke.json) remains separate.

The server release remains separately sealed. `activate.sh` changes into that server directory. To inspect this concise repository, return to your clone and consult [CODE.md](CODE.md). Do not equate a newly built checkout with an already accepted service binary; each new deployment needs its own preflight/smoke evidence.

## Supplementary PA/SAGE ablation

The prepared-server four-arm AE request **passed** on 2026-09-25 at 00:55:23 UTC+8. It ran from 2026-09-24 23:55:55 for 59 min 28 s, including smoke, initialization and full workers. All four smoke and four full workers were fresh processes, exited normally and recorded zero monitor query errors. Each full arm completed one seed-0 epoch: 1,207,179 examples and 1,179 updates, without validation/test. The service exited successfully and GPU 2 was released.

[Current results](ABLATION.md): 194.63/196.12/160.38/103.69 s, 1.8770×. The main PA three-model protocol stays at 20 epochs with validation/test.

## Four-arm AE command

The extension is installed on the prepared server. To read existing accepted results without starting another run, use `digit-ae results PA sage --action ablation`.

```bash
digit-ae ablation PA sage
digit-ae status PA sage --action ablation
digit-ae results PA sage --action ablation
digit-ae logs PA sage --action ablation
digit-ae stop PA sage --action ablation
```

A new request runs fresh GIDS → adjacency-only +GR → ++NS → DiGiT smoke workers, then one full training epoch per arm. GPU 2, data paths and protocol are fixed. Requests share the main AE GPU/NVMe exclusion lock and reject busy resources. The service survives SSH disconnects. `PASS` requires closed accepted reports and successful service exit; the CLI never substitutes an earlier success for a later failure.

The extension source is in `tools/ablation/` and targets a separately provisioned runtime snapshot with read-only data mounts. It does not provide a standalone four-arm run from the concise checkout. The main two-arm `run.sh` interface and submitted `ae-pa-v1` remain separate.

## Four-arm monitoring

The service uses the original author external `nvidia-smi` monitor: a 5-second subprocess query timeout and a 0.5-second wait after each query. The first successful sample is required before a worker starts. Later query errors are recorded while sampling continues; final acceptance requires zero query errors, normal worker and monitor exits, valid ownership, and resource/report checks. There is no live heartbeat or phase-gap gate. Available-host-memory admission is 192 GiB; cache capacities and training candidates are unchanged.
