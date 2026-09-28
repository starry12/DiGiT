# IG/SAGE performance comparison

Final speedup: **2.02×**, the **maximum observed paired speedup across five rounds**.
This is an accepted author result. Installation and a fresh AE replay are separately
verified; no fresh AE replay is claimed here.

## Reproduction on the prepared server

After the administrator installs the IG extension:

```bash
source /srv/digit-ae/activate.sh
digit-ae performance IG sage
digit-ae status IG sage --action performance
digit-ae results IG sage --action performance
digit-ae logs IG sage --action performance
digit-ae stop IG sage --action performance
```

To inspect the author reference without running training:
`digit-ae results IG sage --action performance --reference`.
It is labeled `AUTHOR_REFERENCE`, never a fresh `PASS`.

Each request runs five paired rounds, ten fresh processes, ordered
G1,D1,D2,G2,G3,D3,D4,G4,G5,D5. Each process warms up for 20 mini-batches
and measures 300 mini-batches, with seed 0 and batch size 1024.
The selected statistic is the largest **same-round** GIDS/DiGiT total-time ratio,
computed at full precision. It is not the ratio of independently chosen runs,
mean performance, stable performance, or evidence of interference-free execution.
No accuracy or full-epoch claim is made. Sampling, feature access and model updates
are real. Preparation and warmup are outside the measured interval.

GIDS retains default CPU scheduling; original DiGiT binds to logical CPU2 after
imports. GPU2, cache settings, host-stage timing, CPU/GPU telemetry and the inherited
NVML monitor remain fixed. Available host memory admission is 320 GiB.
A request takes exclusive AE/author resource locks, rejects busy devices, and never
writes the prepared raw SSD region. There are no automatic retries.

Default results show only final speedup. Complete raw evidence remains in the
server output directory; `--json` is available for explicit inspection. `PASS`
requires all ten accepted workers and successful service completion. Failed or
incomplete latest requests are not replaced by earlier successes.

The fixed service adapters and original source dependencies are in
`tools/ig_performance/`. They target the separately installed, hash-bound prepared
runtime with read-only input mounts. This is not a claim of an independently
validated from-scratch IG build from the concise checkout. The installer performs
CPU/import checks without launching training. The PA release and submitted
`ae-pa-v1` remain unchanged.
