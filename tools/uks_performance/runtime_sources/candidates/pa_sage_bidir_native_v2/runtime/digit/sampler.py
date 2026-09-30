"""DGL BlockSampler implementing DiGiT outermost group-aware sampling."""

from collections import defaultdict, deque
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import dgl
import numpy as np
import torch
from dgl import EID, NID
from dgl.dataloading import BlockSampler

from .artifacts import ArtifactBundle, ArtifactValidationError, load_artifact_bundle
from .outer_selection import (
    ExpandedEdges, SampledFrontier, UniformNodewisePolicy, select_units,
)

try:
    from . import DiGiTSamplerCUDA as _cuda_extension
except ImportError:
    _cuda_extension = None


DIGIT_STORAGE_ROW = "digit_storage_row"
DIGIT_STORAGE_IS_GROUP = "digit_storage_is_group"
DIGIT_SAMPLED_GROUPS = "digit_sampled_groups"
DIGIT_SAMPLED_NODES = "digit_sampled_nodes"
_DIGIT_EDGE_STORAGE_ROW = "_digit_edge_storage_row"
_DIGIT_EDGE_IS_GROUP = "_digit_edge_is_group"


def feature_rows_from_blocks(blocks: Sequence[dgl.DGLGraph]) -> torch.Tensor:
    """Return storage rows aligned with the outermost block's logical sources."""

    if not blocks:
        raise ValueError("blocks cannot be empty")
    if DIGIT_STORAGE_ROW not in blocks[0].srcdata:
        raise KeyError("outermost block does not contain DiGiT storage rows")
    rows = blocks[0].srcdata[DIGIT_STORAGE_ROW]
    if rows.shape != blocks[0].srcdata[NID].shape:
        raise RuntimeError("DiGiT storage rows are not aligned with input nodes")
    return rows


class DiGiTNeighborSampler(BlockSampler):
    """Use standard sampling in inner layers and group units only at input.

    Version 1 supports homogeneous inbound uniform sampling without
    replacement. The returned DGL blocks contain only logical node IDs. The
    selected physical feature row for each input node is attached to
    ``blocks[0].srcdata[DIGIT_STORAGE_ROW]``.
    """

    def __init__(
        self,
        fanouts: Sequence[int],
        artifact: Union[ArtifactBundle, str, Path],
        *,
        edge_dir: str = "in",
        prob: Optional[str] = None,
        replace: bool = False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        cuda_mode: str = "auto",
        random_seed: int = 0,
        group_selection: bool = True,
        metadata_mode: str = "gpu",
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        if not fanouts:
            raise ValueError("fanouts cannot be empty")
        self.fanouts = [int(value) for value in fanouts]
        if any(value < -1 for value in self.fanouts):
            raise ValueError("fanouts must be non-negative or -1")
        if edge_dir != "in":
            raise NotImplementedError("DiGiT v1 sampler supports edge_dir='in' only")
        if prob is not None:
            raise NotImplementedError("DiGiT v1 sampler currently supports uniform sampling")
        if replace:
            raise NotImplementedError("DiGiT v1 sampler samples without replacement")
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        if cuda_mode not in ("auto", "required", "disabled"):
            raise ValueError("cuda_mode must be 'auto', 'required', or 'disabled'")
        if cuda_mode == "required" and _cuda_extension is None:
            raise RuntimeError("DiGiTSamplerCUDA is required but is not built")
        self.cuda_mode = cuda_mode
        if metadata_mode not in ('gpu', 'cpu_eid', 'gpu_i32', 'gpu_i32_uva_eid64'):
            raise ValueError('metadata_mode must be gpu, cpu_eid or gpu_i32')
        self.metadata_mode = metadata_mode
        self._original_cpu_csc = None
        self.group_selection = bool(group_selection)
        if not self.group_selection and cuda_mode == "required":
            raise ValueError(
                "cuda_mode='required' is incompatible with disabled group selection"
            )
        self.random_seed = int(random_seed)
        self._cuda_call_counter = 0
        self._cuda_metadata = {}
        self._cuda_graph_identity = None
        self.bundle = (
            artifact
            if isinstance(artifact, ArtifactBundle)
            else load_artifact_bundle(artifact, mmap_mode="r", verify_checksums=True)
        )
        self.num_nodes = self.bundle.num_nodes
        self.num_groups = self.bundle.num_groups
        self.group_size = self.bundle.group_size
        self._indptr = self.bundle.arrays["reordered_indptr"]
        self._indices = self.bundle.arrays["reordered_indices"]
        self._group_members = self.bundle.arrays["group_members"]
        self._group_storage_base = self.bundle.arrays["group_storage_base"]
        self._supernode_to_group = self.bundle.arrays["supernode_to_group"]
        self._node_to_primary = self.bundle.arrays["node_to_primary_row"]
        self.outer_policy = UniformNodewisePolicy()

    @property
    def cuda_extension_available(self) -> bool:
        return _cuda_extension is not None

    def _cuda_enabled_for(self, seed_nodes: torch.Tensor, fanout: int) -> bool:
        if self.cuda_mode == "disabled":
            return False
        if seed_nodes.device.type != "cuda" or fanout <= 0:
            if self.cuda_mode == "required":
                raise RuntimeError("CUDA sampler requires CUDA seeds and a positive fanout")
            return False
        if _cuda_extension is None:
            if self.cuda_mode == "required":
                raise RuntimeError("DiGiTSamplerCUDA is required but is not built")
            return False
        if fanout > int(_cuda_extension.MAX_FANOUT):
            if self.cuda_mode == "required":
                raise ValueError("fanout exceeds the CUDA extension limit")
            return False
        if fanout < self.group_size:
            if self.cuda_mode == "required":
                raise ValueError("CUDA fanout cannot be smaller than group_size")
            return False
        return True

    @staticmethod
    def _numpy_to_cuda(array: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.tensor(np.asarray(array), dtype=torch.int64, device=device)

    def _ensure_cuda_metadata(
        self, graph: dgl.DGLGraph, device: torch.device
    ) -> Mapping[str, torch.Tensor]:
        device_key = str(device)
        graph_identity = id(graph)
        if self._cuda_graph_identity not in (None, graph_identity):
            self._cuda_metadata.clear()
        self._cuda_graph_identity = graph_identity
        if device_key in self._cuda_metadata:
            return self._cuda_metadata[device_key]

        graph.create_formats_()
        original_indptr, original_indices, original_eids = graph.adj_tensors("csc")
        if self.metadata_mode == 'gpu_i32_uva_eid64':
            from .compact_gpu_ids import check_all_ids,upload_ids
            if graph.device.type!='cpu' or not graph.is_pinned():
                raise ValueError('gpu_i32_uva_eid64 requires a pinned CPU CSC graph')
            if getattr(_cuda_extension,'INT32_METADATA_UVA64_API',0)!=1:
                raise RuntimeError('Candidate sampler CUDA extension is missing')
            original=(original_indptr,original_indices,original_eids)
            if any(t.dtype!=torch.int64 or not t.is_contiguous() or not t.is_pinned() for t in original):
                raise ValueError('Original CSC must be contiguous pinned int64, including EIDs')
            if graph.num_edges()!=self.bundle.manifest['dataset']['num_edges']:
                raise ValueError('Graph differs from bidirectional overlay')
            arrays=dict(reorganized_indices=self._indices,group_members=self._group_members,
                group_storage_base=self._group_storage_base,supernode_to_group=self._supernode_to_group,
                node_to_primary=self._node_to_primary)
            check_all_ids(arrays.values())
            metadata={name:upload_ids(array,device) for name,array in arrays.items()}
            metadata['reorganized_indptr']=self._numpy_to_cuda(self._indptr,device)
            metadata.update(original_indptr=original_indptr,original_indices=original_indices,original_eids=original_eids)
            self._cuda_metadata[device_key]=metadata
            return metadata
        if self.metadata_mode == 'gpu_i32':
            from .compact_gpu_ids import check_all_ids,upload_ids
            if graph.device.type!='cpu':raise ValueError('gpu_i32 requires CPU/UVA source graph')
            if getattr(_cuda_extension,'INT32_METADATA_API',0)!=1:
                raise RuntimeError('rebuild DiGiTSamplerCUDA for gpu_i32')
            arrays=dict(reorganized_indices=self._indices,group_members=self._group_members,
                group_storage_base=self._group_storage_base,supernode_to_group=self._supernode_to_group,
                node_to_primary=self._node_to_primary,original_indices=original_indices.numpy(),
                original_eids=original_eids.numpy())
            check_all_ids(arrays.values())
            metadata={name:upload_ids(array,device) for name,array in arrays.items()}
            metadata['reorganized_indptr']=self._numpy_to_cuda(self._indptr,device)
            metadata['original_indptr']=original_indptr.to(device=device,dtype=torch.int64)
            self._cuda_metadata[device_key]=metadata
            return metadata
        if self.metadata_mode == 'cpu_eid':
            if graph.device.type != 'cpu':
                raise ValueError('cpu_eid requires the original DGL graph on CPU/UVA')
            if getattr(_cuda_extension, 'CPU_EID_SELECTION_API', 0) != 1:
                raise RuntimeError('rebuild DiGiTSamplerCUDA for cpu_eid support')
            self._original_cpu_csc = (original_indptr.numpy(), original_indices.numpy(), original_eids.numpy())
        metadata = {
            "reorganized_indptr": self._numpy_to_cuda(self._indptr, device),
            "reorganized_indices": self._numpy_to_cuda(self._indices, device),
            "group_members": self._numpy_to_cuda(self._group_members, device),
            "group_storage_base": self._numpy_to_cuda(self._group_storage_base, device),
            "supernode_to_group": self._numpy_to_cuda(self._supernode_to_group, device),
            "node_to_primary": self._numpy_to_cuda(self._node_to_primary, device),
        }
        if self.metadata_mode == 'gpu':
            metadata.update(original_indptr=original_indptr.to(device=device,dtype=torch.int64),
                            original_indices=original_indices.to(device=device,dtype=torch.int64),
                            original_eids=original_eids.to(device=device,dtype=torch.int64))
        metadata = {name: tensor.contiguous() for name, tensor in metadata.items()}
        self._cuda_metadata[device_key] = metadata
        return metadata

    def _validate_graph(self, graph: dgl.DGLGraph) -> None:
        if (self.outer_policy.policy_id != "uniform_nodewise_v1"
                or self.outer_policy.budget_scope != "destination"):
            raise NotImplementedError("9C1 only enables the frozen uniform node-wise policy")
        if not graph.is_homogeneous:
            raise TypeError("DiGiTNeighborSampler v1 requires a homogeneous DGL graph")
        if graph.num_nodes() != self.num_nodes:
            raise ArtifactValidationError(
                "DGL graph node count does not match the DiGiT artifact"
            )

    def _sample_unit_indices(
        self, graph_ids: np.ndarray, fanout: int, device: torch.device
    ) -> np.ndarray:
        """Compatibility helper; production CPU uses the scoped interfaces below."""
        candidates = self.outer_policy.candidates(
            -1, 0, graph_ids, self.num_nodes, self.group_size)
        return self.select_outer_units(candidates, fanout, device).indices

    def generate_outer_candidates(self, owner):
        start, end = int(self._indptr[owner]), int(self._indptr[owner + 1])
        return self.outer_policy.candidates(
            owner, start, self._indices[start:end], self.num_nodes, self.group_size)

    def select_outer_units(self, candidates, fanout, device, choices=None):
        return select_units(candidates, fanout, device, choices)

    def expand_outer_units(self, candidates, selected, seed_position, expanded):
        """Append occurrences in draw/member order, preserving Python row ties."""
        for graph_id in candidates.graph_ids[selected.indices]:
            graph_id = int(graph_id)
            if graph_id < self.num_nodes:
                expanded.sources.append(graph_id)
                expanded.destinations.append(candidates.owner)
                expanded.node_counts[seed_position] += 1
                continue
            supernode_slot = graph_id - self.num_nodes
            if not 0 <= supernode_slot < self.num_groups:
                raise ArtifactValidationError("reorganized CSC contains an invalid supernode")
            group_id = int(self._supernode_to_group[supernode_slot])
            members = np.asarray(self._group_members[group_id], dtype=np.int64)
            base = int(self._group_storage_base[group_id])
            for offset, member in enumerate(members):
                member = int(member)
                expanded.sources.append(member)
                expanded.destinations.append(candidates.owner)
                expanded.preferred_group_rows.setdefault(member, base + offset)
            expanded.group_counts[seed_position] += 1
            expanded.node_counts[seed_position] += self.group_size

    def _resolve_original_eids(
        self,
        graph: dgl.DGLGraph,
        seed_nodes: torch.Tensor,
        selected_sources: Sequence[int],
        selected_destinations: Sequence[int],
    ) -> torch.Tensor:
        """Map selected logical pairs to distinct valid original edge IDs."""

        all_sources, all_destinations, all_eids = graph.in_edges(seed_nodes, form="all")
        queues: Mapping[Tuple[int, int], deque] = defaultdict(deque)
        for source, destination, eid in zip(
            all_sources.detach().cpu().tolist(),
            all_destinations.detach().cpu().tolist(),
            all_eids.detach().cpu().tolist(),
        ):
            queues[(int(source), int(destination))].append(int(eid))

        selected_eids = []
        for source, destination in zip(selected_sources, selected_destinations):
            queue = queues[(int(source), int(destination))]
            if not queue:
                raise ArtifactValidationError(
                    "artifact edge ({}, {}) is absent from the DGL graph".format(
                        source, destination
                    )
                )
            selected_eids.append(queue.popleft())
        return torch.as_tensor(
            selected_eids, dtype=graph.idtype, device=seed_nodes.device
        )

    def _group_frontier(
        self, graph: dgl.DGLGraph, seed_nodes: torch.Tensor, fanout: int
    ) -> Tuple[dgl.DGLGraph, Mapping[int, int], torch.Tensor, torch.Tensor]:
        seeds_cpu = seed_nodes.detach().cpu().numpy().astype(np.int64, copy=False)
        if seeds_cpu.size and (seeds_cpu.min() < 0 or seeds_cpu.max() >= self.num_nodes):
            raise IndexError("logical seed node is outside the artifact")
        expanded = ExpandedEdges.empty(seeds_cpu.size)
        for seed_position, owner in enumerate(seeds_cpu):
            candidates = self.generate_outer_candidates(owner)
            selected = self.select_outer_units(candidates, fanout, seed_nodes.device)
            self.expand_outer_units(candidates, selected, seed_position, expanded)
        return self.build_outer_frontier(graph, seed_nodes, expanded)

    def build_outer_frontier(self, graph, seed_nodes, expanded):
        """Resolve original EIDs only after ordered expansion (multigraph queues)."""
        sources = torch.as_tensor(
            expanded.sources, dtype=graph.idtype, device=seed_nodes.device
        )
        destinations = torch.as_tensor(
            expanded.destinations, dtype=graph.idtype, device=seed_nodes.device
        )
        frontier = dgl.graph(
            (sources, destinations),
            num_nodes=self.num_nodes,
            idtype=graph.idtype,
            device=seed_nodes.device,
        )
        frontier.edata[EID] = self._resolve_original_eids(
            graph, seed_nodes, expanded.sources, expanded.destinations
        )
        return (
            frontier,
            expanded.preferred_group_rows,
            torch.as_tensor(expanded.group_counts, dtype=torch.int64, device=seed_nodes.device),
            torch.as_tensor(expanded.node_counts, dtype=torch.int64, device=seed_nodes.device),
        )

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

        sample_cuda = (_cuda_extension.sample_group_aware_i32_uva64 if self.metadata_mode=='gpu_i32_uva_eid64' else
                       _cuda_extension.sample_group_aware_i32 if self.metadata_mode=='gpu_i32'
                       else _cuda_extension.sample_group_aware)
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
            from .cpu_eids import resolve_first
            resolved = resolve_first(*self._original_cpu_csc,
                seeds.detach().cpu().numpy(),
                output_sources.detach().cpu().numpy().reshape(num_seeds,fanout))
            if (resolved == -2).any():
                raise ArtifactValidationError('selected edge absent from original CSC')
            output_eids = torch.as_tensor(resolved.reshape(-1),device=seeds.device)

        valid = torch.nonzero(output_sources >= 0, as_tuple=True)[0]
        sources = output_sources.index_select(0, valid)
        destinations = seeds.index_select(0, torch.div(valid, fanout, rounding_mode="floor"))
        frontier = dgl.graph(
            (sources, destinations),
            num_nodes=self.num_nodes,
            idtype=torch.int64,
            device=seeds.device,
        )
        frontier.edata[EID] = output_eids.index_select(0, valid)
        frontier.edata[_DIGIT_EDGE_STORAGE_ROW] = output_rows.index_select(0, valid)
        frontier.edata[_DIGIT_EDGE_IS_GROUP] = output_is_group.index_select(0, valid).to(torch.bool)
        return frontier, output_group_counts, output_node_counts

    def _attach_storage_rows(
        self, block: dgl.DGLGraph, preferred_group_rows: Mapping[int, int]
    ) -> None:
        logical_nodes = block.srcdata[NID].detach().cpu().numpy().astype(np.int64, copy=False)
        storage_rows = np.asarray(self._node_to_primary[logical_nodes], dtype=np.int64).copy()
        is_group = np.zeros(logical_nodes.size, dtype=bool)
        for position, logical_node in enumerate(logical_nodes):
            preferred = preferred_group_rows.get(int(logical_node))
            if preferred is not None:
                storage_rows[position] = preferred
                is_group[position] = True
        block.srcdata[DIGIT_STORAGE_ROW] = torch.as_tensor(
            storage_rows, dtype=torch.int64, device=block.device
        )
        block.srcdata[DIGIT_STORAGE_IS_GROUP] = torch.as_tensor(
            is_group, dtype=torch.bool, device=block.device
        )

    def _attach_cuda_storage_rows(
        self, block: dgl.DGLGraph, metadata: Mapping[str, torch.Tensor]
    ) -> None:
        if self.metadata_mode in ('gpu_i32','gpu_i32_uva_eid64'):
            from .output_fusion import annotate
            local_sources, _ = block.edges(order="eid")
            rows, flags = annotate(block.srcdata[NID], metadata["node_to_primary"],
                                   local_sources, block.edata[_DIGIT_EDGE_STORAGE_ROW],
                                   block.edata[_DIGIT_EDGE_IS_GROUP])
            block.srcdata[DIGIT_STORAGE_ROW] = rows
            block.srcdata[DIGIT_STORAGE_IS_GROUP] = flags
            return
        logical_nodes = block.srcdata[NID].to(torch.int64)
        primary_rows = metadata["node_to_primary"][logical_nodes].to(torch.int64)
        preferred_rows = torch.full_like(primary_rows, torch.iinfo(torch.int64).max)
        local_sources, _ = block.edges(order="eid")
        group_edges = block.edata[_DIGIT_EDGE_IS_GROUP]
        # No boolean compaction: ordinary edges contribute the neutral amin value.
        group_rows = torch.where(group_edges, block.edata[_DIGIT_EDGE_STORAGE_ROW],
                                 torch.iinfo(torch.int64).max)
        preferred_rows.scatter_reduce_(
            0, local_sources, group_rows, reduce="amin", include_self=True)
        has_group_row = preferred_rows != torch.iinfo(torch.int64).max
        block.srcdata[DIGIT_STORAGE_ROW] = torch.where(
            has_group_row, preferred_rows, primary_rows
        )
        block.srcdata[DIGIT_STORAGE_IS_GROUP] = has_group_row

    def sample_inner_frontier(self, graph, seed_nodes, fanout):
        """Unmodified native sampling, also used by the standard-DGL control."""
        return graph.sample_neighbors(
            seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
            replace=self.replace, output_device=self.output_device)

    def sample_outer_frontier(self, graph, seed_nodes, fanout):
        """Dispatch one backend without materializing per-unit GPU objects."""
        if not self.group_selection:
            frontier = self.sample_inner_frontier(graph, seed_nodes, fanout)
            return SampledFrontier(
                frontier, torch.zeros(seed_nodes.shape, dtype=torch.int64,
                                      device=seed_nodes.device),
                frontier.in_degrees(seed_nodes).to(torch.int64), "primary", {})
        if self._cuda_enabled_for(seed_nodes, fanout):
            # Candidate/selection/expansion stay fused in the unchanged CUDA ABI.
            frontier, groups, nodes = self._cuda_group_frontier(graph, seed_nodes, fanout)
            return SampledFrontier(frontier, groups, nodes, "uniform_cuda_legacy_v1")
        frontier, rows, groups, nodes = self._group_frontier(graph, seed_nodes, fanout)
        return SampledFrontier(frontier, groups, nodes, "uniform_python_legacy_v1", rows)

    @staticmethod
    def build_block(frontier, seed_nodes):
        block = dgl.to_block(frontier, seed_nodes)
        if EID in frontier.edata:
            block.edata[EID] = frontier.edata[EID]
        elif EID in block.edata:
            del block.edata[EID]
        return block

    def annotate_outer_storage(self, graph, block, sampled):
        """FeatureAccessPlan stays aligned with the block's logical source order."""
        if sampled.storage_profile == "uniform_cuda_legacy_v1":
            self._attach_cuda_storage_rows(block, self._ensure_cuda_metadata(graph, block.device))
        elif sampled.storage_profile in ("uniform_python_legacy_v1", "primary"):
            self._attach_storage_rows(block, sampled.preferred_group_rows)
        else:
            raise ValueError("unsupported storage profile: " + sampled.storage_profile)
        block.dstdata[DIGIT_SAMPLED_GROUPS] = sampled.group_counts.to(block.device)
        block.dstdata[DIGIT_SAMPLED_NODES] = sampled.node_counts.to(block.device)

    def sample_blocks(self, graph, seed_nodes, exclude_eids=None):
        self._validate_graph(graph)
        if exclude_eids is not None:
            raise NotImplementedError("exclude_eids is not supported by DiGiT v1")
        if isinstance(seed_nodes, Mapping):
            raise TypeError("DiGiTNeighborSampler v1 does not support heterogeneous seeds")
        if not torch.is_tensor(seed_nodes):
            seed_nodes = torch.as_tensor(seed_nodes, dtype=graph.idtype, device=graph.device)
        if seed_nodes.dtype != graph.idtype:
            seed_nodes = seed_nodes.to(dtype=graph.idtype)

        output_nodes = seed_nodes
        blocks = []
        reversed_fanouts = list(reversed(self.fanouts))
        for reverse_layer, fanout in enumerate(reversed_fanouts):
            outermost = reverse_layer == len(reversed_fanouts) - 1
            if outermost:
                sampled = self.sample_outer_frontier(graph, seed_nodes, fanout)
                frontier = sampled.graph
            else:
                frontier = self.sample_inner_frontier(graph, seed_nodes, fanout)
            block = self.build_block(frontier, seed_nodes)
            if outermost:
                self.annotate_outer_storage(graph, block, sampled)
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks
