"""Historical four-head GAT with explicit dimensions and DGL attention defaults."""
import torch
from ae.papers.runtime.models import GAT as HistoricalGAT
from training.gat.common import cfg,require

class GAT(HistoricalGAT):
    def forward(self,blocks,x):
        require(len(blocks)==len(self.layers),'GAT requires exactly three sampled blocks')
        require(x.ndim==2 and x.shape[1]==128,'Wrong GAT input dimension')
        rows=x.shape[0]
        for block in blocks:
            require(block.is_block and block.num_src_nodes()==rows and block.device==x.device,'GAT block row/device mismatch')
            rows=block.num_dst_nodes()
        return super().forward(blocks,x)

def make_model(device='cpu',dtype=torch.float32):
    p=cfg();model=GAT(**p['model_config']).to(device=device,dtype=dtype)
    require(sum(v.numel() for v in model.parameters())==p['model_parameter_count'],'GAT parameter count changed')
    for layer,inp,out in zip(model.layers,(128,512,512),(128,128,172)):
        require(layer._num_heads==4 and layer._in_src_feats==layer._in_dst_feats==inp and layer._out_feats==out,'GAT head/width contract changed')
        require(layer.fc.bias is None and layer.bias is not None and layer.has_explicit_bias and layer.res_fc is None and layer.activation is None,'GAT projection/bias/residual changed')
        require(layer.feat_drop.p==layer.attn_drop.p==0 and layer.leaky_relu.negative_slope==.2 and not layer._allow_zero_in_degree,'GAT attention defaults changed')
    return model
