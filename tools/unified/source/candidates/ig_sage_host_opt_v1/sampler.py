"""Reuse one stable valid-edge index for every frontier output column."""
import torch
import dgl
from uva_sampler import UVANeighborSampler
from digit.sampler import _DIGIT_EDGE_STORAGE_ROW,_DIGIT_EDGE_IS_GROUP


def compact_columns(sources,rows,flags,eids,seeds,fanout):
    # nonzero preserves the same increasing slot order as boolean indexing.
    # CUDA boolean indexing independently computes this list for each column.
    positions=(sources>=0).nonzero(as_tuple=False).flatten()
    destinations=seeds.index_select(0,torch.div(positions,fanout,rounding_mode='floor'))
    return (sources.index_select(0,positions),destinations,
            rows.index_select(0,positions),flags.index_select(0,positions),
            eids.index_select(0,positions))


class SharedIndexSampler(UVANeighborSampler):
    def _cuda_group_frontier(self,graph,seed_nodes,fanout):
        metadata=self._ensure_cuda_metadata(graph,seed_nodes.device)
        invocation_seed=self.random_seed+self._cuda_call_counter
        sources,rows,flags,eids,groups,nodes=metadata.sample(seed_nodes,fanout,invocation_seed)
        self._cuda_call_counter+=1
        src,dst,physical,is_group,original_eids=compact_columns(sources,rows,flags,eids,seed_nodes,fanout)
        frontier=dgl.graph((src,dst),num_nodes=self.num_nodes,idtype=torch.int64,device=seed_nodes.device)
        frontier.edata[dgl.EID]=original_eids
        frontier.edata[_DIGIT_EDGE_STORAGE_ROW]=physical
        frontier.edata[_DIGIT_EDGE_IS_GROUP]=is_group.to(torch.bool)
        return frontier,groups,nodes


def sampler_type(variant):
    if variant not in ('legacy','affinity','compact','combined'):
        raise ValueError('Unknown variant')
    return SharedIndexSampler if variant in ('compact','combined') else UVANeighborSampler
