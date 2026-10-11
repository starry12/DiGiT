"""Guard the frozen model interface without changing parameter names or math."""
import dgl
import torch
from main_common import D, S, ROOT, module, read, require

LEGACY = module('digit_frozen_gcn_gat_models', ROOT / 'ae/papers/runtime/models.py')

def config(name):
    models = read(D / 'contract.json')['models']
    if name not in models:
        raise ValueError('Unsupported model: ' + str(name))
    return dict(models[name])

def validate_blocks(blocks, x, cfg, logical=False):
    require(len(blocks) == cfg['num_layers'], 'Expected exactly three blocks; zip truncation is invalid')
    require(x.ndim == 2 and x.shape[1] == cfg['in_feats'], 'Wrong feature rank/width')
    rows = x.shape[0]
    previous = None
    for i, block in enumerate(blocks):
        require(block.is_block and len(block.canonical_etypes) == 1, 'Expected homogeneous sampling block')
        require(block.num_src_nodes() == rows, 'Block source rows do not align at layer %d' % i)
        require(block.device == x.device, 'Block/features on different devices')
        rows = block.num_dst_nodes()
        require(rows > 0, 'Empty destination batch')
        if logical:
            require(dgl.NID in block.srcdata and dgl.NID in block.dstdata, 'Missing logical NIDs')
            src, dst = block.srcdata[dgl.NID], block.dstdata[dgl.NID]
            require(torch.equal(src[:rows], dst), 'Destination NIDs must prefix source NIDs')
            if previous is not None:
                require(torch.equal(previous, src), 'Inter-block logical NID ordering mismatch')
            previous = dst

class CheckedForward:
    check_logical_ids = False

    def forward(self, blocks, x):
        validate_blocks(blocks, x, self.interface_config, self.check_logical_ids)
        logits = super().forward(blocks, x)
        require(tuple(logits.shape) == (blocks[-1].num_dst_nodes(), self.interface_config['num_classes']),
                'Wrong output logits shape')
        return logits

class CheckedGCN(CheckedForward, LEGACY.GCN):
    pass

class CheckedGAT(CheckedForward, LEGACY.GAT):
    pass

def make_model(name, device='cpu', dtype=torch.float32, checked=True, logical=False):
    cfg = config(name)
    cls = {'gcn': CheckedGCN, 'gat': CheckedGAT}[name] if checked else getattr(LEGACY, name.upper())
    model = cls(**cfg).to(device=device, dtype=dtype)
    if checked:
        model.interface_config = cfg
        model.check_logical_ids = logical
    return model

def make_optimizer(model):
    options = read(D / 'contract.json')['optimizer']
    options['betas'] = tuple(options['betas'])
    return torch.optim.Adam(model.parameters(), **options)
