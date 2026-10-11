"""Same AE int64 CSC and EIDs, copied to anonymous memory before pinning."""
def load_pinned_csc(data,nodes,edges):
    from ae.igb.bounded_io import load_csc
    graph,report=load_csc(*[data/('original_'+k+'.npy') for k in ('indptr','indices','eids')],
        nodes,host_cap=8*(nodes+1+2*edges))
    tensors=graph.adj_tensors('csc')
    return graph,[x.numpy() for x in tensors]
