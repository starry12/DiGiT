"""Accepted final dense Graph and incremental sampler, only mean normalization differs."""
import torch
import torch.nn.functional as F
from candidates.pa_sage_dense_graph_v1.model import DenseGraphSAGE, prepare_model
from .normalization import fused_mean


class MeanGraphSAGE(DenseGraphSAGE):
    def __init__(self, *args, mean_mode='legacy', **kwargs):
        if mean_mode not in ('legacy', 'fused'):
            raise ValueError('Unknown mean normalization mode')
        super().__init__(*args, **kwargs)
        self.mean_mode = mean_mode

    def forward(self, blocks, x):
        if self.mean_mode == 'legacy':
            return super().forward(blocks, x)
        if not hasattr(self, '_dense_stats') or self._dense_stats['mode'] != 'graph':
            raise RuntimeError('Prepare accepted dense graph before fused mean training')
        h = x
        for index, (layer, block) in enumerate(zip(self.layers, blocks)):
            last = index == len(self.layers) - 1
            mean = fused_mean(block, h)
            destination = h[:block.num_dst_nodes()]
            if last and self.training and torch.is_grad_enabled() and block.num_dst_nodes() == self._dense_stats['capture_batch_size']:
                if mean.device != self._dense_device or mean.dtype != torch.float32 or not mean.requires_grad or not destination.requires_grad:
                    raise RuntimeError('Captured dense input contract changed')
                h = self._dense_graphed(mean, destination)
                self._dense_stats['graph_forwards'] += 1
            else:
                neighbor = layer.fc_neigh(mean)
                h = layer.fc_self(destination) + neighbor
                if last:
                    self._dense_stats['eager_forwards'] += 1
            if not last:
                h = F.relu(self.dropout(h))
        return h
