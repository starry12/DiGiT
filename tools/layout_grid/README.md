# Prepared-server layout-grid extension

See [protocol, commands and results](../../docs/LAYOUT_GRID.md).
`controller.py` runs 15 fresh point controllers over a separately installed, immutable
runtime snapshot. `point_runner.py` is the accepted point controller with an explicit
isolated-root bootstrap and shared locks moved to the whole-grid parent. Its worker,
sampler, I/O, cache and validation code are unchanged. No arbitrary privileged paths
or protocol options are accepted from the reviewer CLI.

`runtime_sources/` contains the exact supplementary grid, layout builder and continuation
sources used by the author run. Their `candidates.*` imports describe the provisioned
runtime namespace. Existing core implementation is documented under `training/sage`,
`runtime/io`, `ae/pa_sage`, `ae/papers` and `third_party/bam`. This directory is a code
and deployment interface, not a standalone native run from a fresh public clone:
original dependency manifests, native binaries, prepared data and private device/pool
bindings are installed separately. Do not run historical source launchers directly.

CPU command/acceptance regression:

```bash
python3 -B tools/layout_grid/tests.py
```

Those tests mock services and results; they do not start systemd, nvidia-smi, CUDA or SSD I/O.
The read-only namespace/import selftest is an additional administrator installation check.
A fresh full AE request still requires its own native acceptance.
