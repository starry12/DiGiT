"""Stage the current clean package plus eight explicit model routes."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

H = Path(__file__).resolve().parent
SOURCE = Path('/home/embed/digit-ae-clean')
P = H / 'reviewer'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if P.exists():
        shutil.rmtree(P)
    shutil.copytree(SOURCE, P, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'results'))
    # Reapply the final change to its recorded base, not to the previously
    # generated README/CLI. This avoids duplicate routes, tests and prose.
    previous = H.parent / 'ae_multimodel_20261006_v1'
    before = json.loads((previous / 'checkout_plan.json').read_text())
    for name in ('README.md','docs/CODE.md','tools/reviewer/cli.py','tools/reviewer/test_cli.py'):
        if sha(SOURCE / name) != before['files'][name]['after']:
            raise RuntimeError('Local model adaptation changed since v1: ' + name)
        with tarfile.open(previous / 'checkout_before.tar') as archive:
            (P / name).write_bytes(archive.extractfile(name).read())
    f = P / 'tools/reviewer/cli.py'
    s = f.read_text().replace("if a.model != 'sage' or selected != 'performance':", "if selected != 'performance':")
    s = s.replace('use performance <dataset> sage;', 'use performance <dataset> <sage|gcn|gat>;')
    s = s.replace("        a.route = a.dataset\n", "        a.route = a.dataset if a.model == 'sage' else a.dataset + '_' + a.model\n")
    s = s.replace("INSPECT =", "HANDLERS.update({dataset + '_' + model: ADMIN / 'multimodel_v2' / (dataset.lower() + '_' + model) / 'cli.py'\n                 for dataset in ('IG', 'UKS', 'UKL', 'CL') for model in ('gcn', 'gat')})\nINSPECT =")
    s = s.replace("    command = ['/usr/bin/python3'", "    if not HANDLERS[a.route].is_file():\n        raise RuntimeError('This model service is not installed on the prepared server yet: ' + a.route)\n    command = ['/usr/bin/python3'")
    f.write_text(s)
    f = P / 'tools/reviewer/test_cli.py'
    s = f.read_text().replace("'performance UKS gat',", "'performance UKS other',")
    s = s.replace("    def test_rejects_unpublished_or_ambiguous_workloads(self):", '''    def test_new_model_routes_and_reference_isolation(self):
        for dataset in ('IG','UKS','UKL','CL'):
            for model in ('gcn','gat'):
                for command in ('performance', *cli.INSPECT):
                    argv = [command,dataset,model] + ([] if command == 'performance' else ['--action','performance'])
                    self.assertEqual(cli.parse(argv).route,dataset+'_'+model)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    cli.parse(['results',dataset,model,'--action','performance','--reference'])

    def test_rejects_unpublished_or_ambiguous_workloads(self):''')
    # Frontend tests exercise routing without requiring a locally installed server.
    s = s.replace("class ReviewerCLI(unittest.TestCase):", "class ReviewerCLI(unittest.TestCase):\n    def setUp(self):\n        mock = patch.object(Path, 'is_file', return_value=True)\n        mock.start()\n        self.addCleanup(mock.stop)\n")
    f.write_text(s)
    f = P / 'README.md'
    s = f.read_text().replace('### GraphSAGE performance: IG / UKS / UKL / CL', '### Model performance: IG / UKS / UKL / CL', 1)
    anchor = '| Experiment | Start | Results |'
    start = s.index('|', s.index('### Model performance:'))
    end = s.index('\n\n', start)
    s = s[:start] + '''| Dataset | GraphSAGE | GCN | GAT |
|---|---|---|---|
| IG | `digit-ae performance IG sage` | `digit-ae performance IG gcn` | `digit-ae performance IG gat` |
| UKS | `digit-ae performance UKS sage` | `digit-ae performance UKS gcn` | `digit-ae performance UKS gat` |
| UKL | `digit-ae performance UKL sage` | `digit-ae performance UKL gcn` | `digit-ae performance UKL gat` |
| CL | `digit-ae performance CL sage` | `digit-ae performance CL gcn` | `digit-ae performance CL gat` |

GCN/GAT adapters passed CPU and synthetic CUDA checks for all eight combinations. Activation on the prepared server and native performance validation are pending. These models have no reference speedups yet. [Model protocol and installation](tools/performance_models/README.md).

Inspect the selected dataset and model with `digit-ae results IG gcn --action performance`; substitute the dataset/model for other comparisons.''' + s[end:]
    f.write_text(s)
    dest = P / 'tools/performance_models'
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(H / 'payload', dest / 'deployment')
    for name in ('prepare.py', 'prepare_reviewer.py', 'install.py', 'install_support.py'):
        if (H / name).exists():
            shutil.copyfile(H / name, dest / name)
    (dest / 'README.md').write_text('''# GCN/GAT short-window models

The four datasets accept `gcn` and `gat` in the performance, status, logs,
results and stop commands after installing the prepared model services.
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
''')
    f = P / 'docs/CODE.md'
    s = f.read_text().replace('| Automatic GPU selection', '| IG/UKS/UKL/CL GCN and GAT adapters | `tools/performance_models/` |\n| Automatic GPU selection')
    f.write_text(s)
    manifest = json.loads((P / 'ARTIFACT_MANIFEST.json').read_text())
    manifest['files'] = {str(p.relative_to(P)): sha(p) for p in sorted(P.rglob('*'))
                         if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'}
    (P / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('Staged reviewer package', sha(P / 'ARTIFACT_MANIFEST.json'))


if __name__ == '__main__':
    main()
