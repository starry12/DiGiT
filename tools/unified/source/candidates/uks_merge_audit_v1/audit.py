import numpy as np

def pairs(rows):
    rows=np.unique(rows)
    low=rows[(rows%4)==0]
    return low[np.isin(low+1,rows)]

def summarize(nodes,local,edge_rows,edge_flags,rows,flags,hot):
    er=edge_rows[edge_flags];el=local[edge_flags]
    u,first=np.unique(er,return_index=True);candidate=pairs(u)
    pos=np.searchsorted(u,candidate);pos1=np.searchsorted(u,candidate+1)
    same=(el[first[pos]]==el[first[pos1]])
    marked=rows[flags];actual=pairs(marked)
    order=np.argsort(rows);sr=rows[order]
    i=order[np.searchsorted(sr,actual)];j=order[np.searchsorted(sr,actual+1)]
    h0=hot[nodes[i]];h1=hot[nodes[j]]
    return dict(feature_rows=len(nodes),group_edge_occurrences=len(er),unique_group_physical_rows=len(u),
        unique_group_logical_nodes=len(np.unique(el)),repeated_group_edge_occurrences=len(er)-len(u),
        candidate_complete_pairs=len(candidate),candidate_same_logical_pairs=int(same.sum()),
        mapped_group_rows=len(marked),mapped_unflagged_rows=int((~flags).sum()),mapped_complete_pairs=len(actual),
        mapped_orphan_group_rows=len(marked)-2*len(actual),
        candidate_pairs_not_retained=len(candidate)-len(actual),
        mapped_pairs_both_cpu_hot=int((h0&h1).sum()),mapped_pairs_one_cpu_hot=int((h0^h1).sum()),
        mapped_pairs_both_cpu_cold=int((~h0&~h1).sum()))

class Audit:
    def __init__(self,sampler):
        from .common import DATA,read
        from candidates.uks_native_v1.binding import protocol
        self.hot=np.zeros(protocol()['nodes'],dtype=np.bool_)
        self.hot[np.load(DATA/'rank/hot_nodes.npy',mmap_mode='r')]=True
        self.records=[];self.batch=0;old=sampler._attach_cuda_storage_rows
        def annotate(block,metadata):
            old(block,metadata);self.batch+=1
            if self.batch<=20:return
            import dgl
            from digit.sampler import DIGIT_STORAGE_ROW,DIGIT_STORAGE_IS_GROUP,_DIGIT_EDGE_STORAGE_ROW,_DIGIT_EDGE_IS_GROUP
            def a(t):return t.detach().cpu().numpy()
            local,_=block.edges(order='eid')
            self.records.append(summarize(a(block.srcdata[dgl.NID]),a(local),a(block.edata[_DIGIT_EDGE_STORAGE_ROW]),a(block.edata[_DIGIT_EDGE_IS_GROUP]),a(block.srcdata[DIGIT_STORAGE_ROW]),a(block.srcdata[DIGIT_STORAGE_IS_GROUP]),self.hot))
        sampler._attach_cuda_storage_rows=annotate
    def finish(self):
        assert self.batch==320 and len(self.records)==300
        return dict(per_batch=self.records,totals={k:sum(r[k] for r in self.records) for k in self.records[0]},timing_valid=False)
