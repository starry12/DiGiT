"""Candidate-local raw adjacency adapter; original NPY reader stays strict."""
from uva_sampler import UVANeighborSampler, UVAMetadata, arrays_from_graph
from .raw_adjacency import RawAdjacencySource


class MigrationSampler(UVANeighborSampler):
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
