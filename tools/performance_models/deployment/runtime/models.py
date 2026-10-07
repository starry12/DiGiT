"""Explicit model contracts; no data loading, CUDA initialization or SSD access."""
import hashlib
import json

DIMENSIONS = {'IG': 1024, 'UKS': 256, 'UKL': 128, 'CL': 128}
LARGE_GAT_MODEL_LIMIT = 2*2**30 + 128*2**20


def contract(dataset, model):
    if dataset not in DIMENSIONS or model not in ('gcn', 'gat'):
        raise ValueError('Expected IG/UKS/UKL/CL and gcn/gat')
    config = dict(in_feats=DIMENSIONS[dataset], h_feats=128, num_classes=19,
                  num_layers=3, dropout=0.2)
    if model == 'gat':
        config['num_heads'] = 4
    return dict(dataset=dataset, model=model, config=config, seed=0,
                optimizer=dict(lr=0.001, weight_decay=0.001),
                fanouts=[10, 5, 5], batch_size=1024, rounds=5,
                warmup_batches=20, measured_batches=300,
                accuracy_evaluated=False, zero_degree_policy='reject',
                attention_execution='checkpointed_sequential_heads_shared_parameters' if model == 'gat' else None,
                semantics='GraphConv norm=both' if model == 'gcn' else
                'GATConv four heads, hidden concatenate, output head mean')


def make_model(dataset, model, device='cpu', dtype=None):
    import torch
    from ae.igb.models import GCN, GAT
    config = contract(dataset, model)['config']
    base = GCN if model == 'gcn' else GAT

    class CheckedModel(base):
        def forward(self, blocks, x):
            if len(blocks) != config['num_layers'] or x.ndim != 2 or x.shape[1] != config['in_feats']:
                raise ValueError('Model layer count or input width mismatch')
            rows = x.shape[0]
            for block in blocks:
                if not block.is_block or block.num_src_nodes() != rows or block.device != x.device:
                    raise ValueError('Sampled block order, device or row count mismatch')
                rows = block.num_dst_nodes()
            if model == 'gcn':
                return super().forward(blocks, x)
            # The historical four-head parameters and equations are unchanged.
            # Separate head graphs avoid a full-width source-gradient temporary
            # in DGL's fused backward. Both systems use this same model path.
            import torch.nn.functional as F
            h = x
            for index, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = headwise_attention(layer, block, h)
                if index < len(self.layers) - 1:
                    h = self.dropout(F.relu(h.flatten(1)))
                else:
                    h = h.mean(1)
            return h

    return CheckedModel(**config).to(device=device, dtype=dtype or torch.float32)


def headwise_attention(layer, graph, x):
    """Exact no-residual GATConv equations, one head at a time."""
    import torch
    import torch.nn.functional as F
    import dgl
    import dgl.function as fn
    from dgl.nn.functional import edge_softmax
    if layer.res_fc is not None or layer.activation is not None or layer.feat_drop.p or layer.attn_drop.p:
        raise ValueError('Unsupported historical GAT configuration')
    if not layer._allow_zero_in_degree and (graph.in_degrees() == 0).any():
        raise dgl.DGLError('Zero-in-degree destination in sampled GAT block')
    outputs = []
    width = layer._out_feats
    for head in range(layer._num_heads):
        def one_head(features, index=head):
            with graph.local_scope():
                projected = F.linear(features, layer.fc.weight[index*width:(index+1)*width]).unsqueeze(1)
                dst = projected[:graph.num_dst_nodes()]
                left = (projected * layer.attn_l[:,index:index+1]).sum(-1).unsqueeze(-1)
                right = (dst * layer.attn_r[:,index:index+1]).sum(-1).unsqueeze(-1)
                graph.srcdata.update(ft=projected, el=left)
                graph.dstdata['er'] = right
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                scores = layer.leaky_relu(graph.edata.pop('e'))
                graph.edata['a'] = edge_softmax(graph, scores)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                return graph.dstdata['ft']
        if torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            outputs.append(checkpoint(one_head, x, use_reentrant=False, preserve_rng_state=False))
        else:
            outputs.append(one_head(x))
    result = torch.cat(outputs, dim=1)
    if layer.has_explicit_bias:
        result = result + layer.bias.view(1, layer._num_heads, width)
    return result


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def identity(dataset, model, source_sha):
    value = contract(dataset, model)
    return dict(value, adapter_sha256=source_sha,
                contract_sha256=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())


def require_identity(report, expected):
    if report.get('model_identity') != expected:
        raise RuntimeError('Wrong dataset/model/adapter identity; SAGE evidence cannot qualify')


def model_memory(dataset, model):
    """Conservative float32 forward/backward/Adam envelope for sampled blocks."""
    c = contract(dataset, model)
    frontier = c['batch_size']
    blocks = []
    for fanout in reversed(c['fanouts']):
        src = frontier * (fanout + 1)
        blocks.insert(0, dict(src=src, dst=frontier, edges=frontier * fanout))
        frontier = src
    widths = [128, 128, 19]
    heads = 4 if model == 'gat' else 1
    inputs = [DIMENSIONS[dataset], 128 * heads, 128 * heads]
    outputs = [w * heads for w in widths]
    params = sum(i * o + (3 if model == 'gat' else 1) * o for i, o in zip(inputs, outputs))
    parts = dict(parameters_adam_grad=params * 32,
                 input_features=blocks[0]['src'] * DIMENSIONS[dataset] * 4,
                 block_indices=sum((b['src'] + b['dst'] + 2 * b['edges']) * 8 for b in blocks))
    if model == 'gat':
        parts['attention_forward_backward'] = sum(
            16 * (b['src'] + b['dst']) * w + 4 * b['edges'] * w
            + 4 * heads * (6 * b['edges'] + 4 * (b['src'] + b['dst']))
            + 16 * b['dst'] * w for b, w in zip(blocks, outputs))
    else:
        parts['normalization_forward_backward'] = sum(
            16 * (b['src'] * i + b['dst'] * o) + 32 * (b['src'] + b['dst'] + b['edges'])
            for b, i, o in zip(blocks, inputs, outputs))
    return dict(components=parts, bytes=(sum(parts.values()) * 6 + 4) // 5,
                blocks=blocks, measured=False)


def cpu_check(dataset, model):
    """Real forward/backward/Adam on shrinking blocks, without GPU/SSD."""
    import torch
    import dgl
    torch.set_num_threads(1)
    c = contract(dataset, model)
    blocks = [dgl.create_block((torch.arange(dst), torch.arange(dst)),
                              num_src_nodes=src, num_dst_nodes=dst)
              for src, dst in ((8, 6), (6, 4), (4, 2))]
    torch.random.default_generator.manual_seed(0)
    net = make_model(dataset, model)
    initial = state_hash(net)
    opt = torch.optim.Adam(net.parameters(), **c['optimizer'])
    x = torch.linspace(-1, 1, 8 * DIMENSIONS[dataset]).reshape(8, -1)
    losses = []
    for _ in range(2):
        pred = net(blocks, x)
        if pred.shape != (2, 19):
            raise RuntimeError('Wrong output shape')
        loss = torch.nn.functional.cross_entropy(pred, torch.tensor([0, 18]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()):
            raise RuntimeError('Nonfinite/missing gradients')
        opt.step()
        losses.append(float(loss.detach()))
    if state_hash(net) == initial or not all(torch.isfinite(p).all() for p in net.parameters()):
        raise RuntimeError('No finite optimizer update')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU selftest unexpectedly initialized CUDA')
    return dict(passed=True, dataset=dataset, model=model, losses=losses,
                parameters=sum(p.numel() for p in net.parameters()),
                cuda_initialized=False, native_acceptance=False, raw_ssd_access=False)
