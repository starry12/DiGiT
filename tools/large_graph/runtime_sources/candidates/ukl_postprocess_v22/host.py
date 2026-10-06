"""One stable node numbering per layer; no per-edge Python loops.

Native RNG, output validation, EID order and first-occurrence physical-row
selection are unchanged. All temporary arrays scale with the sampled frontier.
"""
from dataclasses import dataclass
import numpy as np
from candidates.ukl_native_sampling_v10 import sampling as S


@dataclass
class Layer:
    targets: np.ndarray
    nodes: np.ndarray
    sources_local: np.ndarray
    destinations_local: np.ndarray
    first_source: np.ndarray
    result: dict


def number_layer(targets, result):
    targets = np.asarray(targets)
    sources = np.asarray(result['src'])
    destinations = np.asarray(result['dst'])
    if (targets.ndim != 1 or sources.ndim != 1 or destinations.shape != sources.shape
            or targets.dtype.kind not in 'iu' or sources.dtype.kind not in 'iu'
            or destinations.dtype.kind not in 'iu' or len(sources) > S.MAX_OUTPUT_EDGES
            or len(targets) > S.MAX_SEEDS):
        raise ValueError('Bounded integer sampled layer required')
    joined = np.concatenate((targets, sources))
    unique, first, inverse = np.unique(joined, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    local = rank[inverse]
    if not np.array_equal(local[:len(targets)], np.arange(len(targets))):
        raise ValueError('Duplicate targets')
    nodes = joined[first[order]].copy()
    u = local[len(targets):].copy()
    # The already computed sorted unique table supplies destination lookup.
    where = np.searchsorted(unique, destinations)
    if np.any(where >= len(unique)) or not np.array_equal(unique[where], destinations):
        raise ValueError('Unknown destination')
    v = rank[where]
    if np.any(v >= len(targets)):
        raise ValueError('Destination outside target prefix')
    first_source = np.full(len(nodes), len(sources), dtype=np.int64)
    np.minimum.at(first_source, u, np.arange(len(sources), dtype=np.int64))
    return Layer(targets.copy(), nodes, u, v, first_source, result)


def storage_rows(layer, graph, grouped):
    addresses = np.array(graph.arrays['primary'][layer.nodes] if grouped else layer.nodes,
                         dtype=np.int64, copy=True)
    present = layer.first_source < len(layer.result['src'])
    addresses[present] = layer.result['rows'][layer.first_source[present]]
    if np.any(addresses < 0) or np.any(addresses >= (graph.storage_rows if grouped else graph.nodes)):
        raise RuntimeError('Block storage rows outside graph')
    return addresses


def make_block(layer):
    import torch, dgl
    block = dgl.create_block((torch.from_numpy(layer.sources_local),
                              torch.from_numpy(layer.destinations_local)),
                             num_src_nodes=len(layer.nodes), num_dst_nodes=len(layer.targets))
    block.srcdata[dgl.NID] = torch.from_numpy(layer.nodes.astype(np.int64, copy=False))
    block.dstdata[dgl.NID] = torch.from_numpy(layer.targets.astype(np.int64, copy=False))
    block.edata[dgl.EID] = torch.from_numpy(layer.result['eid'].astype(np.int64, copy=False))
    return block


class Sampler(S.Sampler):
    def layers(self, roots, batch=0, audit=None):
        roots = np.asarray(roots)
        if (roots.ndim != 1 or roots.dtype.kind not in 'iu' or not 0 < len(roots) <= 1024
                or np.any(roots < 0) or np.any(roots >= self.native.graph.nodes)
                or len(np.unique(roots)) != len(roots) or type(batch) is not int or batch < 0
                or self.seed + batch * 3 + 2 > 2**64 - 1):
            raise ValueError('Unique roots, batch and seed must satisfy 1024-root bounds')
        seeds = np.array(roots, dtype=np.int32, copy=True)
        layers = []
        for layer in (2, 1, 0):
            fanout = (10, 5, 5)[layer]
            raw = self.native.sample(seeds, fanout, grouped=self.grouped and layer == 0,
                                     seed=self.seed + batch * 3 + layer, packed=self.grouped)
            if audit is not None:
                audit(layer, seeds.copy(), fanout, self.grouped and layer == 0, self.grouped, raw)
            result = S.compact(raw, fanout)
            numbered = number_layer(seeds, result)
            layers.insert(0, numbered)
            seeds = numbered.nodes
        return seeds.copy(), layers

    def sample_blocks(self, roots, batch=0, *, layers=None):
        import torch
        from candidates.ukl_sage_compact_v1.sampler import STORAGE_ROW
        nodes, numbered = self.layers(roots, batch) if layers is None else layers
        blocks = []
        for i, layer in enumerate(numbered):
            block = make_block(layer)
            if i == 0:
                block.srcdata[STORAGE_ROW] = torch.from_numpy(storage_rows(layer, self.native.graph, self.grouped))
            blocks.append(block)
        return torch.from_numpy(nodes.astype(np.int64, copy=True)), torch.from_numpy(np.array(roots, dtype=np.int64, copy=True)), blocks
