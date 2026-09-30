"""Shared graph semantics: stable non-self edges, then one self edge per node."""
POLICY = 'remove_self_loops_then_append_one_per_node_v1'


def normalize_dgl_graph(graph):
    import dgl
    return dgl.add_self_loop(dgl.remove_self_loop(graph))


def check_training_source(bundle, graph, args):
    if args.data != 'OGB':
        return
    from pathlib import Path
    from . import large_preprocess as lp
    contract = bundle.manifest.get('metadata', {}).get('graph_source')
    if not contract or contract.get('policy') != POLICY:
        raise ValueError('OGB DiGiT requires a graph-source contract with normalized self loops')
    if graph.num_nodes() != contract['num_nodes'] or graph.num_edges() != contract['num_edges']:
        raise ValueError('training graph size disagrees with normalized artifact')
    current = lp.identity(Path(args.path) / 'edge_index.npy')
    expected = contract['original_edges']
    for key in ('sha256', 'shape', 'dtype', 'fortran_order'):
        if current[key] != expected[key]:
            raise ValueError('training edge source differs from normalized artifact')
