"""Capture only the fixed-shape final two Linear operators and their backward.

Parameter objects remain registered solely under the original SAGE layers.
The private wrapper holds those same objects without registering extra state in
the parent model. DGL aggregation, dropout, loss and Adam remain eager.
"""
import time
import torch
from torch import nn
import torch.nn.functional as F
import dgl.ops
from candidates.pa_sage_512b_latency_v1.model import FunctionalMeanSAGE, functional_layer


class FinalDense(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.fc_neigh = layer.fc_neigh
        self.fc_self = layer.fc_self

    def forward(self, mean, destination):
        neighbor = self.fc_neigh(mean)
        return self.fc_self(destination) + neighbor


class DenseGraphSAGE(FunctionalMeanSAGE):
    def prepare_dense(self, mode, batch_size=1024, warmup_iters=3):
        if mode not in ('eager', 'graph') or batch_size <= 0 or warmup_iters != 3:
            raise ValueError('Expected eager/graph, positive batch size, and three warmups')
        if hasattr(self, '_dense_stats'):
            raise RuntimeError('Prepare once before iteration; runtime recapture is forbidden')
        params = tuple(self.parameters())
        if not self.training or any(p.device.type != 'cuda' or p.dtype != torch.float32 for p in params):
            raise ValueError('Prepare FP32 CUDA training model before iteration')
        if any(p.grad is not None for p in params):
            raise ValueError('Capture before the first training gradient')
        if self.layers[-1].fc_neigh.in_features != 128 or self.layers[-1].fc_self.in_features != 128:
            raise ValueError('This candidate captures the frozen 128-wide final layer')
        memory_before = torch.cuda.memory_allocated(params[0].device)
        identities = [(id(p), p.data_ptr()) for p in params]
        keys = tuple(self.state_dict())
        initial = [p.detach().clone() for p in params]
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state(params[0].device))
        wrapper = FinalDense(self.layers[-1])
        device = params[0].device
        args = tuple(torch.zeros(batch_size, 128, device=device, requires_grad=True) for _ in range(2))
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        if mode == 'graph':
            wrapper = torch.cuda.make_graphed_callables(wrapper, args, num_warmup_iters=warmup_iters)
        else:
            # Match the API's dense-only forward/backward warmups, without Adam
            # updates or dropout/RNG consumption. The live eager path stays original.
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                for _ in range(warmup_iters):
                    out = wrapper(*args)
                    grads = torch.autograd.grad(out, args + tuple(wrapper.parameters()),
                                                grad_outputs=torch.empty_like(out))
                del out, grads
        torch.cuda.synchronize(device)
        preparation_seconds = time.perf_counter() - begin
        if [(id(p), p.data_ptr()) for p in self.parameters()] != identities or tuple(self.state_dict()) != keys:
            raise RuntimeError('Graph capture replaced parameters or changed state_dict')
        if any(p.grad is not None for p in params) or not all(torch.equal(a, p) for a, p in zip(initial, params)):
            raise RuntimeError('Preparation modified parameters/gradients')
        if not torch.equal(rng[0], torch.get_rng_state()) or not torch.equal(rng[1], torch.cuda.get_rng_state(device)):
            raise RuntimeError('Dense preparation consumed RNG')
        # Avoid nn.Module registration here: keep original parameter/state_dict paths.
        object.__setattr__(self, '_dense_graphed', wrapper if mode == 'graph' else None)
        # Only the capture closure retains its static inputs; audit clones and
        # eager sample inputs are not part of the live model's memory footprint.
        del args, initial, rng
        self._dense_stats = dict(mode=mode, capture_batch_size=batch_size, prepared=True,
            capture_count=int(mode == 'graph'), graph_forwards=0, eager_forwards=0,
            warmup_iters=warmup_iters, preparation_seconds=preparation_seconds,
            retained_allocation_bytes=max(0, torch.cuda.memory_allocated(device) - memory_before),
            parameter_identity_preserved=True, initial_parameters_unchanged=True, rng_preserved=True,
            optimizer_identity_preserved=False, scope='final two Linear operators and add; forward and backward')
        self._dense_device = device
        return self.dense_summary()

    def dense_summary(self):
        return dict(self._dense_stats)

    def forward(self, blocks, x):
        if not hasattr(self, '_dense_stats'):
            raise RuntimeError('Call prepare_dense before training')
        if self._dense_stats['mode'] == 'eager':
            self._dense_stats['eager_forwards'] += 1
            return super().forward(blocks, x)
        h = x
        for index, (layer, block) in enumerate(zip(self.layers, blocks)):
            last = index == len(self.layers) - 1
            if last and self.training and torch.is_grad_enabled() and block.num_dst_nodes() == self._dense_stats['capture_batch_size']:
                mean = dgl.ops.copy_u_mean(block, h)
                destination = h[:block.num_dst_nodes()]
                if mean.device != self._dense_device or mean.dtype != torch.float32 or not mean.requires_grad or not destination.requires_grad:
                    raise RuntimeError('Captured dense input contract changed')
                h = self._dense_graphed(mean, destination)
                self._dense_stats['graph_forwards'] += 1
            else:
                h = functional_layer(layer, block, h)
                if last:
                    self._dense_stats['eager_forwards'] += 1
            if not last:
                h = self.dropout(h)
                h = F.relu(h)
        return h


def prepare_model(model, optimizer, mode, batch_size=1024):
    before = [id(p) for group in optimizer.param_groups for p in group['params']]
    model_ids = [id(p) for p in model.parameters()]
    if before != model_ids or optimizer.state:
        raise RuntimeError('Prepare with untouched Adam over original parameters')
    model.prepare_dense(mode, batch_size=batch_size)
    after = [id(p) for group in optimizer.param_groups for p in group['params']]
    if before != after or before != [id(p) for p in model.parameters()] or optimizer.state:
        raise RuntimeError('Preparation altered optimizer references or state')
    model._dense_stats['optimizer_identity_preserved'] = True
    return model.dense_summary()
