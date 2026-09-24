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

## Supplementary author ablation

[Updated PA/SAGE ablation](ABLATION.md) is curated existing author evidence: 194.45/194.76/161.73/104.50 s, with aligned RevPR hot sets for the first three arms and adjacency-only GR. It ran through author services, not the reviewer self-service interface. The standard two-arm run command does not reproduce a four-arm ablation. The main reviewer protocol stays at 20 epochs with validation/test.

## Four-arm AE command

The prepared-server extension adds these commands after administrator installation:

```bash
digit-ae ablation PA sage
digit-ae status PA sage --action ablation
digit-ae results PA sage --action ablation
digit-ae logs PA sage --action ablation
digit-ae stop PA sage --action ablation
```

Each request runs four fresh smoke workers, then GIDS → adjacency-only +GR → ++NS → DiGiT for one complete training epoch each, without validation/test. It reuses prepared data but does not reuse historical performance reports. GPU 2, data locations, units and protocol are fixed; busy requests fail instead of queueing. The global GPU/NVMe lock is shared with the main AE services. Starting and stopping require only two exact administrator-installed sudo permissions; no arbitrary privileged command, configuration, GPU or output path is accepted.

The service uses a separate root-owned snapshot and read-only prepared-data mounts. Existing PA/IG commands and the submitted sealed release remain separate. The extension sources are under `tools/ablation/`; they target this prepared server and its matching administrator-provisioned runtime snapshot, not a standalone ablation run from the concise checkout. The standard two-arm `run.sh` interface remains unchanged.

The four-arm extension is installed and has been launched through the AE account. Reproduction is **incomplete and paused at the author's request** until a quieter server window. GIDS/+GR smoke evidence was reused and ++NS smoke passed in the latest request; DiGiT smoke stopped after a 13.03-second NVML query exceeded the strict heartbeat limit. No full epoch ran in that AE request. These partial results do not replace the accepted [author table](ABLATION.md).

The deployed service now uses a 192 GiB available-host-memory admission threshold. Preparation delays are recorded; strict monitoring covers model/metadata initialization and all training. This changes neither cache capacities nor the training protocol. The server-side monitoring/admission repairs are later than the extension baseline published under `tools/ablation/`; the existing author receipt retains its original publication-time extension metadata. The submitted tag remains unchanged.
