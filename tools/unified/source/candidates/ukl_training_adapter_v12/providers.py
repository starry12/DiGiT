"""Injectable CPU feature bridge, only for bounded fixtures.

Logical hot-node membership never depends on a selected replica's row address.
The production pinned CPU/GPU cache and SSD backend must be wired separately.
"""
import numpy as np
import torch


class FixtureCacheProvider:
    def __init__(self, hot_ids, hot_features, read_rows):
        ids = np.asarray(hot_ids)
        if (ids.dtype != np.int64 or ids.ndim != 1 or len(ids) > 4096
                or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1])
                or hot_features.device.type != 'cpu' or hot_features.dtype != torch.float32
                or hot_features.shape != (len(ids), 128)
                or not torch.isfinite(hot_features).all().item()):
            raise ValueError('Bounded sorted logical hot set and CPU features required')
        self.hot_ids = ids.copy()
        self.hot_features = hot_features.clone().detach()
        self.read_rows = read_rows
        self.logical_requests = self.cpu_hits = self.row_reads = 0

    def __call__(self, request):
        ids, rows = request.logical_ids, request.storage_rows
        if len(ids) > 4096:
            raise ValueError('CPU fixture provider bound')
        # Also checks row range and integer width before handing addresses out.
        request.byte_offsets()
        positions = np.searchsorted(self.hot_ids, ids)
        hit = positions < len(self.hot_ids)
        hit[hit] &= self.hot_ids[positions[hit]] == ids[hit]
        out = torch.empty((len(ids),128), dtype=torch.float32)
        out[torch.from_numpy(hit)] = self.hot_features[torch.from_numpy(positions[hit])]
        if np.any(~hit):
            value = self.read_rows(rows[~hit].copy())
            if (not isinstance(value,torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.float32 or value.shape != (int((~hit).sum()),128)
                    or not torch.isfinite(value).all().item()):
                raise ValueError('Bad row-reader result')
            out[torch.from_numpy(~hit)] = value
        self.logical_requests += len(ids)
        self.cpu_hits += int(hit.sum()); self.row_reads += int((~hit).sum())
        return out
