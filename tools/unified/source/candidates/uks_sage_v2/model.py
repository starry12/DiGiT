"""Untuned UKS SAGE model contract; imported only for a scheduled model check/run."""
from .common import cfg

def model_config():
    p=cfg();return dict(in_feats=p['feature_dim'],h_feats=p['hidden'],num_classes=p['classes'],num_layers=p['layers'],dropout=p['dropout'])
def make_model(device='cpu'):
    from ae.igb.models import SAGE
    return SAGE(**model_config()).to(device)
def optimizer(model):
    import torch
    p=cfg()['optimizer'];return torch.optim.Adam(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
