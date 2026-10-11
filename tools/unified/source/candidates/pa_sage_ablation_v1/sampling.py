"""Layout-only GR: preserve DGL sampling; attach physical primary rows on GPU."""
import dgl
import torch
from digit.sampler import DIGIT_STORAGE_ROW, DIGIT_STORAGE_IS_GROUP

class LayoutOnlySampler:
    def __init__(self, fanouts, bundle, device):
        self.standard = dgl.dataloading.NeighborSampler(fanouts, replace=False)
        self.primary_array = bundle.arrays['node_to_primary_row']
        self.device = device
        self.primary = None

    def prepare(self):
        # Account for this upload in metadata setup, outside epoch timing.
        if self.primary is not None:
            raise RuntimeError('Primary map already prepared')
        self.primary = torch.tensor(self.primary_array,dtype=torch.int64,device=self.device)

    def sample_blocks(self, graph, seeds):
        if self.primary is None:
            raise RuntimeError('Prepare the persistent primary map before sampling')
        inputs, outputs, blocks = self.standard.sample_blocks(graph, seeds)
        blocks[0].srcdata[DIGIT_STORAGE_ROW] = self.primary[inputs]
        blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP] = torch.zeros_like(inputs, dtype=torch.bool)
        return inputs, outputs, blocks
