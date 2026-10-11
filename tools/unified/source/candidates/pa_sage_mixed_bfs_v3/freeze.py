"""Bind AE graph/optimizer/traces over the frozen optimized three-arm predecessor."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Preserve existing manifest')
    source=predecessor.verify();binding=read(OUT/'inputs.json');check_inputs(binding)
    old=read(predecessor.HERE/'provenance.json');p=read(HERE/'protocol.json')
    prep=read(binding['prepared']);traces={}
    ae_protocol=ROOT/'candidates/pa_sage_bfs_main_v1/protocol.json';ae=read(ae_protocol)
    require(ae['graph']['nodes']==p['nodes'] and ae['graph']['edges']==p['edges'],'AE graph differs')
    require({k:ae['optimizer']['kwargs'][k] for k in ('lr','weight_decay')}==p['optimizer'],'AE optimizer differs')
    dependencies=[ae_protocol,Path(binding['prepared']),Path(binding['overlay_prior_inputs']),
        ROOT/'ae/pa_sage/trace_cache.py',ROOT/'evaluation/digit/eval_trace.py',
        ROOT/'candidates/pa_sage_bfs_decay_v1/mixing.py',
        ROOT/'candidates/pa_sage_bidir_native_v2/overlay.py',ROOT/'candidates/pa_sage_direction_v2/graph.py',Path(old['canonical_order']['path'])]
    for split in ('valid','test'):
        path=ROOT/'results/pa_sage_direction_20260921_v2/bidirectional'/(split+'_trace')
        m=read(path/'manifest.json')
        require(m['graph']==dict(num_nodes=p['nodes'],num_edges=p['edges']),'Wrong AE trace extent')
        csc=m['source']['graph']['csc_sha256']
        for key in ('indptr','indices'):
            require(csc[key]==prep['graph'][key]['array_sha256']==ae['graph']['csc_sha256'][key],'AE trace adjacency differs')
        traces[split]=dict(path=str(path),sha256=sha(path/'manifest.json'));dependencies.append(path/'manifest.json')
    provenance=dict(old,traces=traces,bfs_scope='Same canonical PA training-induced BFS; AE bidirectional graph for training and evaluation',
        accuracy_scope='AE bidirectional uniform traces, CPU features; outside training cache',
        alignment=dict(graph=ae['graph'],optimizer=p['optimizer'],lr_schedule=p['lr_schedule']),
        retained_scope='Same CPU hot sets/capacity, GPU cache, 512B/mixed paths, overlap/dense optimizations and BFS64; not a full replacement of AE I/O configuration')
    write(HERE/'provenance.json',provenance)
    files=list(HERE.glob('*.py'))+[HERE/n for n in ('protocol.json','provenance.json','identity_registry.json','README.md')]
    write(HERE/'manifest.json',dict(predecessor_sha256=source,inputs_sha256=sha(OUT/'inputs.json'),
        files={str(f.relative_to(HERE)):sha(f) for f in files},dependencies={str(f):sha(f) for f in dependencies}))
    print('Frozen '+verify())


if __name__=='__main__':main()
