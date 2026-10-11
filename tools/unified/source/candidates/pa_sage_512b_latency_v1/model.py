"""Same mean-SAGE parameters/operators, fewer generic DGL graph-frame operations."""
import dgl.ops
import torch.nn.functional as F
from models import SAGE


def functional_layer(layer, block, features):
    # All supported widths use mean(X) then W_neigh, matching original SAGEConv.
    mean = dgl.ops.copy_u_mean(block, features)
    neighbor = layer.fc_neigh(mean)
    return layer.fc_self(features[:block.num_dst_nodes()]) + neighbor


class FunctionalMeanSAGE(SAGE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.layers:
            if (layer._aggre_type != 'mean' or layer.feat_drop.p != 0
                    or layer.activation is not None or layer.norm is not None
                    or layer._in_src_feats > layer._out_feats):
                raise ValueError('Functional candidate only supports the frozen mean-SAGE branch')

    def forward(self, blocks, x):
        h = x
        for index, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = functional_layer(layer, block, h)
            if index != len(self.layers)-1:
                h = self.dropout(h)
                h = F.relu(h)
        return h
