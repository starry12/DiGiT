from main_common import *
import gzip
N=111059956; E=1726745828; K=11105992
FREEZE=P
SRC=Path(digit_paths.locations()['papers_csc'])
def splits(name):
    desc = read(FREEZE/'protocol.json')['dataset']['splits'][name]
    assert sha(desc['path']) == desc['sha256']
    with gzip.open(desc['path'], 'rt') as f: ids = np.fromiter((int(x.strip()) for x in f if x.strip()), dtype=np.int64)
    assert len(ids) == desc['count'] and len(np.unique(ids)) == len(ids)
    return ids

def graph_without_labels():
    import torch, dgl
    from digit.graph_source import normalize_dgl_graph
    source = read(FREEZE/'protocol.json')['dataset']['source_contract']['original_edges']
    assert sha(source['path']) == source['sha256']
    # Same COO and normalization as OGBDGLDataset, with no feature/label reads.
    edges = np.load(source['path'], mmap_mode='r')
    assert edges.shape == (1615685872, 2)
    g = dgl.graph((torch.from_numpy(np.asarray(edges[:,0]).copy()),
                   torch.from_numpy(np.asarray(edges[:,1]).copy())), num_nodes=N)
    g = normalize_dgl_graph(g).formats('csc')
    g.create_formats_()
    assert g.num_nodes() == N and g.num_edges() == E
    gp, gi, ge = g.adj_tensors('csc')
    assert np.array_equal(gp.numpy(), arr(SRC/'csc/original_indptr.npy'))
    assert np.array_equal(gi.numpy(), arr(SRC/'csc/original_indices.npy'))
    assert len(ge) == E
    g._graph.pin_memory_()
    return g
