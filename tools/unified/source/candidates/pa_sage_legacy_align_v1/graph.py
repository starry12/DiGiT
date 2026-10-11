"""Reuse directed CSC; edge IDs are explicitly local CSC positions."""
import numpy as np


def load_graph(binding, nodes, edges, pin=True):
    import torch
    import dgl
    arrays = [np.load(binding['csc'][k], mmap_mode='c') for k in ('indptr', 'indices')]
    tensors = [torch.from_numpy(a) for a in arrays]
    graph = dgl.graph(('csc', (tensors[0], tensors[1], torch.empty(0, dtype=torch.int64))),
                      num_nodes=nodes, idtype=torch.int64).formats('csc')
    if graph.num_nodes() != nodes or graph.num_edges() != edges:
        raise RuntimeError('Directed graph size mismatch')
    if pin:
        graph._graph.pin_memory_()
        if not graph.is_pinned():
            raise RuntimeError('UVA graph registration failed')
    return graph, arrays


def verify_blocks(blocks, arrays, fanouts):
    import dgl
    ptr, idx = arrays
    for b, fanout in zip(blocks, fanouts):
        u, v = b.edges(order='eid')
        src = b.srcdata[dgl.NID][u].cpu().numpy()
        dst = b.dstdata[dgl.NID][v].cpu().numpy()
        eids = b.edata[dgl.EID].cpu().numpy()
        if not (np.all(eids >= 0) and np.all(eids < len(idx))):
            raise RuntimeError('Sampled edge ID outside directed CSC')
        if not (np.array_equal(src, idx[eids]) and
                np.array_equal(dst, np.searchsorted(ptr, eids, side='right') - 1)):
            raise RuntimeError('Sampled edge endpoints differ from directed CSC')
        owner = b.dstdata[dgl.NID].cpu().numpy()
        expected = np.minimum(ptr[owner + 1] - ptr[owner], fanout)
        if not np.array_equal(b.in_degrees().cpu().numpy(), expected):
            raise RuntimeError('Neighbor sample cardinality mismatch')
        if len(np.unique(eids)) != len(eids):
            raise RuntimeError('Repeated edge despite sampling without replacement')
    return True
