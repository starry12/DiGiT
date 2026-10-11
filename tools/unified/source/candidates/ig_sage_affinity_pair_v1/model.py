import torch
from ae.igb.models import SAGE,GCN,GAT
from candidates.ig_sage_affinity_pair_v1.common import cfg,require
def model_config(name):
    p=cfg();require(name in p['models'],'Unknown model');kw=dict(in_feats=p['feature_dim'],h_feats=p['hidden'],num_classes=p['classes'],num_layers=p['layers'],dropout=p['dropout'])
    if name=='gat':kw['num_heads']=p['gat_heads']
    return kw
def make_model(name,device='cpu',dtype=torch.float32):
    return dict(sage=SAGE,gcn=GCN,gat=GAT)[name](**model_config(name)).to(device=device,dtype=dtype)
def optimizer(model):
    k=dict(cfg()['optimizer']['kwargs']);k['betas']=tuple(k['betas']);return torch.optim.Adam(model.parameters(),**k)
