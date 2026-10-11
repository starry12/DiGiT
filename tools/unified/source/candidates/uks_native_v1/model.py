"""UKS 256D three-layer SAGE with inherited, explicitly disclosed optimizer choices."""
from .common import require,heavy_gate


def create(p,device='cpu'):
    if device!='cpu':heavy_gate()
    else:require(p.get('fixture') and p['nodes']<=4096,'CPU model execution is bounded to fixtures')
    from candidates.pa_sage_cache_policy_v1.training import seed_cpu,model_hash
    from ae.igb.models import SAGE
    import torch
    seed_cpu(p['seed'])
    net=SAGE(in_feats=256,h_feats=p['hidden'],num_classes=p['classes'],num_layers=p['layers'],dropout=p['dropout']).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=p['optimizer']['lr'],weight_decay=p['optimizer']['weight_decay'])
    initial=model_hash(net);seed_cpu(p['seed']);net.train()
    return net,opt,initial
