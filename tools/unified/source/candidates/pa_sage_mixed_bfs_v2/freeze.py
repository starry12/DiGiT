"""Bind implementation, accepted predecessor and existing BFS/evaluation evidence."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Preserve existing manifest')
    setup();reference_hash=reference.verify()
    previous=ROOT/'results/pa_sage_bfs_decay_20261009_v1/runs/20261009-232414-4187589'
    prior=read(previous/'inputs.json');prep=read(previous/'bfs_preparation.json')
    require(read(previous/'full_summary.json')['passed'] and prep['passed'],'Previous BFS was not accepted')
    canonical=previous/'orders/canonical.npy'
    require(sha(canonical)==prior['files'][str(canonical)]['sha256'],'Canonical order changed')
    mixed_run=reference.OUT/'runs/20261010-043706-130722'
    require(read(mixed_run/'summary.json')['passed'],'Mixed-read predecessor was not accepted')
    traces={};binding=read(OUT/'inputs.json')
    dependencies=[ROOT/'candidates/pa_sage_bfs_decay_v1/mixing.py',ROOT/'ae/pa_sage/trace_cache.py',
        ROOT/'evaluation/digit/eval_trace.py',canonical,previous/'bfs_preparation.json',previous/'full_summary.json',
        mixed_run/'summary.json']
    gids_report=gids_reference.OUT/'native/20260926-141410-2926144/full_gids/report.json'
    require(read(gids_report)['passed'] and read(gids_report)['binary_sha256']==sha(GIDS_BINARY),'GIDS predecessor binary mismatch')
    dependencies.extend([gids_reference.HERE/'manifest.json',gids_report,GIDS_BINARY])
    for split in ('valid','test'):
        path=ROOT/'results/pa_sage_direction_20260921_v2/directed'/(split+'_trace')
        m=read(path/'manifest.json');graph_source=m['source']['graph_source']
        require(m['graph']==dict(num_nodes=111059956,num_edges=1726745828),'Wrong directed trace')
        require(graph_source['original_edges']['sha256']==prep['source_edges_sha256'],'BFS and trace source graph differ')
        for key,filename in [('indptr','original_indptr.npy'),('indices','original_indices.npy')]:
            require(graph_source['csc'][filename]['sha256']==binding['files'][binding['csc'][key]]['sha256'],'Evaluation graph differs from training')
        traces[split]=dict(path=str(path),sha256=sha(path/'manifest.json'))
        dependencies.append(path/'manifest.json')
    write(HERE/'provenance.json',dict(canonical_order=dict(path=str(canonical),sha256=sha(canonical)),
        bfs_source_edges_sha256=prep['source_edges_sha256'],traces=traces,
        gids_reference=dict(report=str(gids_report),binary=str(GIDS_BINARY),binary_sha256=sha(GIDS_BINARY),
            historical_epoch_seconds=read(gids_report)['timing']['e2e_seconds'],path='original SAGE, standard sampling, synchronous bounded 512B reads, RevPR'),
        bfs_scope='Canonical undirected training-induced BFS ranking from the same PA source; training remains directed',
        accuracy_scope='Fixed directed standard-uniform traces; original CPU features; no training-cache effects'))
    files=list(HERE.glob('*.py'))+[HERE/'protocol.json',HERE/'provenance.json',HERE/'README.md']
    write(HERE/'manifest.json',dict(reference_sha256=reference_hash,files={str(p.relative_to(HERE)):sha(p) for p in files},
        dependencies={str(p):sha(p) for p in dependencies}))
    print('Frozen '+verify(),flush=True)


if __name__=='__main__':main()
