"""Feature bridge for device blocks; existing checks retained outside Python edge loops."""
import numpy as np
import torch
import dgl
from candidates.ukl_training_adapter_v12.adapter import FeatureRequest,request_from_blocks as host_request
from candidates.ukl_sage_compact_v1.sampler import STORAGE_ROW


def request_from_blocks(arm,graph,inputs,targets,blocks,roots):
    if blocks and blocks[0].device.type=='cpu':return host_request(arm,graph,inputs,targets,blocks,roots)
    if arm not in ('gids','digit') or len(blocks)!=3 or inputs.device.type!='cpu' or targets.device.type!='cpu':
        raise ValueError('Expected CPU endpoint IDs and three GPU blocks')
    if not np.array_equal(targets.numpy(),roots):raise ValueError('Root order changed')
    for i,b in enumerate(blocks):
        src,dst=b.srcdata[dgl.NID],b.dstdata[dgl.NID]
        if not torch.equal(src[:len(dst)],dst):raise ValueError('SAGE destination prefix mismatch')
        if i and not torch.equal(blocks[i-1].dstdata[dgl.NID],src):raise ValueError('Block chain mismatch')
    if not torch.equal(inputs,blocks[0].srcdata[dgl.NID].cpu()) or not torch.equal(targets,blocks[-1].dstdata[dgl.NID].cpu()):
        raise ValueError('Block endpoints differ from sampled IDs')
    rows=blocks[0].srcdata[STORAGE_ROW].cpu().numpy().copy();ids=inputs.numpy().astype(np.int64,copy=True)
    bound=graph.nodes if arm=='gids' else graph.storage_rows
    if (rows.dtype!=np.int64 or rows.shape!=ids.shape or np.any(ids<0) or np.any(ids>=graph.nodes)
            or len(np.unique(ids))!=len(ids) or np.any(rows<0) or np.any(rows>=bound)):
        raise ValueError('Feature IDs or rows out of range')
    if arm=='gids' and not np.array_equal(ids,rows):raise ValueError('GIDS feature row identity')
    ids.setflags(write=False);rows.setflags(write=False)
    return FeatureRequest(arm,ids,rows,bound)
