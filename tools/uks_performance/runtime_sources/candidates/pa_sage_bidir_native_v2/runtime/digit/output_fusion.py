"""Storage annotation preserving minimum group row and primary fallback semantics."""
import torch
from . import DiGiTOutputCUDA


def annotate(nodes, primary, local_sources, edge_rows, group_edges):
    tensors=(nodes,primary,local_sources,edge_rows,group_edges)
    if any(t.device!=nodes.device or t.ndim!=1 or not t.is_contiguous() for t in tensors):
        raise ValueError('output fusion requires contiguous 1D tensors on one device')
    if (not nodes.is_cuda or nodes.dtype!=torch.int64 or local_sources.dtype!=torch.int64
            or edge_rows.dtype!=torch.int64 or group_edges.dtype!=torch.bool
            or primary.dtype not in (torch.int32,torch.int64)):
        raise ValueError('output fusion requires CUDA and bound ID/flag dtypes')
    if not (local_sources.numel()==edge_rows.numel()==group_edges.numel()):
        raise ValueError('output fusion edge arrays differ in length')
    # ID bounds and nonnegative rows are established by the verified sampler/block.
    # Do not add device-to-host bounds checks or synchronization to this hot path.
    with torch.cuda.device(nodes.device):
        rows=torch.empty_like(nodes)
        flags=torch.empty(nodes.numel(),dtype=torch.bool,device=nodes.device)
        DiGiTOutputCUDA.annotate(nodes.data_ptr(),primary.data_ptr(),primary.element_size()*8,
            local_sources.data_ptr(),edge_rows.data_ptr(),group_edges.data_ptr(),
            rows.data_ptr(),flags.data_ptr(),nodes.numel(),local_sources.numel(),
            torch.cuda.current_stream(nodes.device).cuda_stream)
    return rows,flags
