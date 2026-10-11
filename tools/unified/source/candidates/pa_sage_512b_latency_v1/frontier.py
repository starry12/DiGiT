"""Known-extent COO construction for outputs of the frozen validated sampler."""
import torch
from dgl.convert import create_from_edges


def coo_frontier(sources, destinations, num_nodes, *, check_bounds=False):
    # Metadata checks do not read a CUDA scalar. Native sampler outputs are int64.
    if not isinstance(num_nodes, int) or num_nodes < 0:
        raise ValueError('Invalid graph extent')
    if any(t.dtype != torch.int64 or t.ndim != 1 or not t.is_contiguous()
           for t in (sources, destinations)):
        raise ValueError('Frontier needs contiguous int64 vectors')
    if sources.device != destinations.device or sources.numel() != destinations.numel():
        raise ValueError('Frontier endpoints differ in device or size')
    # This cold-path oracle is for tests/unknown input. Runtime range validity comes
    # from the bound graph, verified native sampler and original-edge smoke checks.
    if check_bounds:
        for tensor in (sources, destinations):
            if tensor.numel() and (tensor.min().item() < 0 or tensor.max().item() >= num_nodes):
                raise ValueError('Frontier endpoint out of bounds')
    return create_from_edges('coo', (sources, destinations), '_N', '_E', '_N',
                             num_nodes, num_nodes, row_sorted=False, col_sorted=False)
