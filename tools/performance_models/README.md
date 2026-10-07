# GCN/GAT short-window models

The four datasets accept `gcn` and `gat` in the performance, status, logs,
results and stop commands. All eight model services were activated on the
prepared server on 2026-10-07 after fresh CPU namespace selftests.
Each model has independent services, latest-request state and output directories.
`--reference` remains available only for the published SAGE references.

All new comparisons retain five paired rounds of 20 warmup plus 300 timed
mini-batches, and select the maximum same-round GIDS/DiGiT ratio. They do not
evaluate accuracy. Native measurements for these eight combinations are pending.
All eight passed independent forward/gradient checks, optimizer updates and
maximum-block CUDA probes on an idle L40. These checks use synthetic sampled
blocks, without loading the datasets or accessing SSD feature storage.
UKS requires separate GIDS and DiGiT correctness smokes before formal workers.
IG/UKL/CL require four real warmup updates with source feature comparisons,
finite losses/gradients/parameters and verified Adam steps before measurement.

| Setting | IG | UKS | UKL | CL |
|---|---:|---:|---:|---:|
| Input width | 1024 | 256 | 128 | 128 |
| Classes | 19 | 19 | 19 | 19 |

All models use three layers, hidden width 128, dropout 0.2, fanout 10/5/5,
batch size 1024 and seed 0. GCN uses GraphConv with symmetric normalization.
GAT uses four heads of width 128, concatenates hidden heads and averages final
heads. Both systems execute attention heads sequentially and checkpoint each
head for backward recomputation, with shared historical parameters, to bound
source-gradient temporaries and retained activations; numerical and
gradient checks compare against the original equations. Zero-degree sampled
destinations are rejected. Adam uses lr=0.001 and
weight_decay=0.001, inherited from each dataset's SAGE protocol, without tuning.

Sampling, graph direction, features, root windows, cache policy, CPU affinity,
NUMA placement, locks and monitoring retain the corresponding final SAGE
runtime. UKS retains its different GIDS and DiGiT root windows. UKL/CL retain
the common GPU postprocessing optimizations on both systems. The immutable
parent source hash and the model-adapter/service hashes are recorded separately;
parent SAGE receipts cannot qualify a new model run. UKL/CL anonymous-memory
loading and ownership protections remain in force. GAT has an explicit additional
activation/attention budget. Before loading the full graph, each UKL/CL GAT
worker must pass a CUDA model-only probe with maximum sampled block sizes,
four Adam updates, a hard allocator cap and 25% plus 256 MiB headroom. The
remaining metadata/cache budget and each arm's inherited safety margin must still fit the
GPU admission. UKL/CL GAT reserves 2 GiB + 128 MiB for model execution and raises
the required free GPU memory by 128 MiB, preserving all inherited safety margins.
Failure stops before graph registration or SSD reads.
Estimates and synthetic model probes do not qualify a native performance run.

`deployment/runtime/` contains the shared model and runtime adapters.
`deployment/<dataset>_<model>/` contains fixed service controllers and units,
including CPU namespace selftests. Model selection cannot supply arbitrary code,
paths or systemd arguments. Existing SAGE services and the frozen PA release
are unchanged.

The prepared administrator installer validates current deployment hashes, takes
the shared experiment locks, installs versioned controls and a reviewer source
view, runs eight fresh CPU namespace selftests, and only then switches the
reviewer pointer. It starts no GPU/SSD performance experiments. The complete
installation bundle and its plan are prepared in the author workspace; the
scripts here document that server-specific deployment rather than providing
a fresh-machine data preparation recipe.

The installer validates each installed unit against its prepared content hash
and the destination mode `0644`, independent of the author workspace umask.
`test_install.py` covers first installation, retry after partial installation,
permission changes, content changes and concurrent reviewer updates. Run it with
`python3 -B tools/performance_models/test_install.py`.

With the artifact Python environment, run the model math/gradient regressions
with `python -B tools/performance_models/test_models.py`. This entrypoint loads
the parent model definitions shipped under `tools/ig_performance/runtime_sources/`.
These CPU tests do not load dataset graphs or access SSD feature storage.
