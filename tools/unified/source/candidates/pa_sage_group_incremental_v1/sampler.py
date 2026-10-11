"""Identical RNG, frontier and EID resolution; switch only group weight maintenance."""
from typing import Tuple
import torch
import dgl
from dgl.base import EID
from digit.sampler import (DiGiTNeighborSampler, _cuda_extension,
    _DIGIT_EDGE_STORAGE_ROW, _DIGIT_EDGE_IS_GROUP, ArtifactValidationError)
from candidates.pa_sage_512b_latency_v1.frontier import coo_frontier
import DiGiTGroupIncrementalCUDA as _group_extension


class IncrementalGroupSampler(DiGiTNeighborSampler):
    def __init__(self, *args, group_mode="legacy", **kwargs):
        if group_mode not in ("legacy", "incremental"):
            raise ValueError("Unknown group mode")
        if kwargs.get("metadata_mode") != "gpu_i32_uva_eid64":
            raise ValueError("This candidate requires frozen gpu_i32_uva_eid64 metadata")
        self.group_mode = group_mode
        super().__init__(*args, **kwargs)

    def _cuda_group_frontier(
        self, graph: dgl.DGLGraph, seed_nodes: torch.Tensor, fanout: int
    ) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        metadata = self._ensure_cuda_metadata(graph, seed_nodes.device)
        seeds = seed_nodes.to(dtype=torch.int64).contiguous()
        num_seeds = int(seeds.numel())
        num_slots = num_seeds * fanout
        output_sources = torch.empty(num_slots, dtype=torch.int64, device=seeds.device)
        output_rows = torch.empty_like(output_sources)
        output_is_group = torch.empty(num_slots, dtype=torch.uint8, device=seeds.device)
        output_eids = torch.empty_like(output_sources)
        output_group_counts = torch.empty(num_seeds, dtype=torch.int64, device=seeds.device)
        output_node_counts = torch.empty_like(output_group_counts)
        stream = torch.cuda.current_stream(seeds.device)
        invocation_seed = self.random_seed + self._cuda_call_counter
        self._cuda_call_counter += 1

        sample_cuda = (_group_extension.sample_group_aware_i32_uva64_incremental
                       if self.group_mode == 'incremental' else
                       _group_extension.sample_group_aware_i32_uva64)
        sample_cuda(
            metadata["reorganized_indptr"].data_ptr(),
            metadata["reorganized_indices"].data_ptr(),
            metadata["group_members"].data_ptr(),
            metadata["group_storage_base"].data_ptr(),
            metadata["supernode_to_group"].data_ptr(),
            metadata["node_to_primary"].data_ptr(),
            metadata["original_indptr"].data_ptr() if self.metadata_mode != 'cpu_eid' else 0,
            metadata["original_indices"].data_ptr() if self.metadata_mode != 'cpu_eid' else 0,
            metadata["original_eids"].data_ptr() if self.metadata_mode != 'cpu_eid' else 0,
            seeds.data_ptr(),
            num_seeds,
            self.num_nodes,
            self.num_groups,
            self.group_size,
            fanout,
            invocation_seed,
            output_sources.data_ptr(),
            output_rows.data_ptr(),
            output_is_group.data_ptr(),
            output_eids.data_ptr(),
            output_group_counts.data_ptr(),
            output_node_counts.data_ptr(),
            stream.cuda_stream,
        )

        if self.metadata_mode == 'cpu_eid':
            from digit.cpu_eids import resolve_first
            resolved = resolve_first(*self._original_cpu_csc,
                seeds.detach().cpu().numpy(),
                output_sources.detach().cpu().numpy().reshape(num_seeds,fanout))
            if (resolved == -2).any():
                raise ArtifactValidationError('selected edge absent from original CSC')
            output_eids = torch.as_tensor(resolved.reshape(-1),device=seeds.device)

        valid = torch.nonzero(output_sources >= 0, as_tuple=True)[0]
        sources = output_sources.index_select(0, valid)
        destinations = seeds.index_select(0, torch.div(valid, fanout, rounding_mode="floor"))
        frontier = coo_frontier(sources, destinations, self.num_nodes)
        frontier.edata[EID] = output_eids.index_select(0, valid)
        frontier.edata[_DIGIT_EDGE_STORAGE_ROW] = output_rows.index_select(0, valid)
        frontier.edata[_DIGIT_EDGE_IS_GROUP] = output_is_group.index_select(0, valid).to(torch.bool)
        return frontier, output_group_counts, output_node_counts

