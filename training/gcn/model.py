"""Exact historical GCN math, with explicit interface and normalization contract."""
import torch
from ae.papers.runtime.models import GCN as HistoricalGCN
from training.gcn.common import cfg,require

class GCN(HistoricalGCN):
    def forward(self,blocks,x):
        require(len(blocks)==len(self.layers),'GCN requires exactly three sampled blocks')
        require(x.ndim==2 and x.shape[1]==128,'Wrong GCN input dimension')
        rows=x.shape[0]
        for block in blocks:
            require(block.is_block and block.num_src_nodes()==rows and block.device==x.device,'GCN block row/device mismatch')
            rows=block.num_dst_nodes()
        return super().forward(blocks,x)

def make_model(device='cpu',dtype=torch.float32):
    p=cfg();model=GCN(**p['model_config']).to(device=device,dtype=dtype)
    require(sum(v.numel() for v in model.parameters())==p['model_parameter_count'],'GCN parameter count changed')
    require(all(layer._norm=='both' and not layer._allow_zero_in_degree and layer.weight is not None and layer.bias is not None for layer in model.layers),'GCN normalization changed')
    return model
