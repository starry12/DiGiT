"""Candidate-local raw adjacency adapter; original NPY reader stays strict."""
from .fast_uva import UVANeighborSampler, UVAMetadata, arrays_from_graph
from .raw_adjacency import RawAdjacencySource


class MigrationSampler(UVANeighborSampler):
    def _attach_cuda_storage_rows(self, block, metadata):
        import torch, dgl, IGPerfNative as backend
        from digit.sampler import (DIGIT_STORAGE_ROW, DIGIT_STORAGE_IS_GROUP,
                                   _DIGIT_EDGE_STORAGE_ROW, _DIGIT_EDGE_IS_GROUP)
        logical = block.srcdata[dgl.NID].contiguous()
        sources, _ = block.edges(order='eid')
        sources = sources.contiguous()
        edge_rows = block.edata[_DIGIT_EDGE_STORAGE_ROW].contiguous()
        edge_flags = block.edata[_DIGIT_EDGE_IS_GROUP].contiguous()
        primary = metadata['node_to_primary']
        if primary.dtype != torch.int32:
            raise ValueError('Optimized IG requires accepted compact i32 primary metadata')
        if not hasattr(self, '_annotation_errors'):
            self._annotation_errors = torch.zeros(1, dtype=torch.int64, device=logical.device)
        rows = torch.empty_like(logical)
        flags = torch.empty_like(logical, dtype=torch.bool)
        stream = torch.cuda.current_stream(logical.device)
        backend.annotate(logical.data_ptr(), primary.data_ptr(), sources.data_ptr(),
                         edge_rows.data_ptr(), edge_flags.data_ptr(), len(logical), len(sources),
                         self.num_nodes, len(self.bundle.arrays['storage_to_node']),
                         rows.data_ptr(), flags.data_ptr(), self._annotation_errors.data_ptr(), stream.cuda_stream)
        # All owners are retained; record allocator lifetimes for this stream.
        for tensor in (logical,sources,edge_rows,edge_flags,rows,flags,self._annotation_errors):
            tensor.record_stream(stream)
        block.srcdata[DIGIT_STORAGE_ROW] = rows
        block.srcdata[DIGIT_STORAGE_IS_GROUP] = flags

    def check_window(self):
        if hasattr(self, '_annotation_errors') and int(self._annotation_errors.item()) != 0:
            raise ValueError('Invalid native storage annotation; window rejected')

    def close(self):
        import torch
        if self._cuda_metadata:
            torch.cuda.synchronize()
        self.check_window()
        super().close()

    def _ensure_cuda_metadata(self, graph, device):
        if self._bound_graph is not None and graph is not self._bound_graph:
            raise ValueError('prototype cannot rebind a graph; create a fresh sampler')
        self._bound_graph = graph
        key = str(device)
        if key not in self._cuda_metadata:
            arrays = arrays_from_graph(graph, self.bundle)
            for target, name in (('reorganized_indptr', 'reordered_indptr'),
                                 ('reorganized_indices', 'reordered_indices')):
                arrays[target] = RawAdjacencySource(arrays[target], self.bundle.manifest['files'][name])
            self._cuda_metadata[key] = UVAMetadata(
                arrays, self.num_nodes, self.num_groups, self.group_size,
                len(self.bundle.arrays['storage_to_node']), device, self.compact,
                self.host_cap, graph=graph, chunk_bytes=self.chunk_bytes)
        return self._cuda_metadata[key]
