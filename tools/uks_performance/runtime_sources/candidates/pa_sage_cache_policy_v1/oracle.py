"""Serial CPU oracle for features and FIFO semantics, never a native benchmark."""
from collections import OrderedDict
import numpy as np
from .common import require
from .mapping import logical_slots, storage_slots
from .metrics import serving


class FeatureOracle:
    def __init__(self, storage_to_node, payload, logical_features, hot_nodes, gpu_bytes,
                 feature_mode='logical_node_real'):
        require(feature_mode == 'logical_node_real', 'Physical-row proxy replicas must not alias a CPU feature')
        self.storage = np.asarray(storage_to_node)
        self.payload = np.asarray(payload)
        features = np.asarray(logical_features)
        require(self.storage.ndim == 1 and self.storage.dtype == np.int64, 'Invalid storage IDs')
        require(self.payload.dtype == features.dtype == np.float32 and
                self.payload.shape == (len(self.storage), 128) and features.shape[1:] == (128,), 'Wrong feature shape/type')
        require(len(self.storage) <= 65536 and len(features) <= 4096, 'CPU oracle only supports bounded small fixtures')
        require(np.isfinite(self.payload).all() and np.isfinite(features).all(), 'Nonfinite fixture features')
        self.slots = logical_slots(np.asarray(hot_nodes), len(features))
        self.row_slots = storage_slots(self.storage, self.slots)
        valid = self.storage >= 0
        require(np.array_equal(self.payload[valid].view(np.uint32), features[self.storage[valid]].view(np.uint32)),
                'Replica feature values differ; cannot alias a logical hot node')
        require(type(gpu_bytes) is int and gpu_bytes >= 0 and gpu_bytes % 4096 == 0, 'Invalid GPU cache budget')
        self.cpu = features[hot_nodes].copy()
        self.capacity_pages = gpu_bytes // 4096
        self.pages = OrderedDict()
        self.requests = self.cpu_hits = self.gpu_hits = self.ssd_rows = self.evictions = 0
        self.address_digest = __import__('hashlib').sha256()

    def fetch(self, logical_ids, storage_rows):
        ids, rows = np.asarray(logical_ids), np.asarray(storage_rows)
        require(ids.ndim == rows.ndim == 1 and ids.dtype == rows.dtype == np.int64 and len(ids) == len(rows), 'Invalid request vectors')
        require(not len(rows) or (rows.min() >= 0 and rows.max() < len(self.storage)), 'Storage address out of range')
        require(not len(ids) or (ids.min() >= 0 and ids.max() < len(self.slots)), 'Logical node out of range')
        require(np.array_equal(self.storage[rows], ids), 'Logical/physical request mapping differs')
        self.address_digest.update(rows.astype('<i8', copy=False).tobytes())
        output = np.empty((len(rows), 128), dtype=np.float32)
        for i, row in enumerate(rows):
            row = int(row)
            page, in_page = divmod(row, 8)
            slot = int(self.row_slots[row])
            self.requests += 1
            if page in self.pages:
                self.gpu_hits += 1
                output[i] = self.pages[page][in_page]
                # FIFO hits do not refresh insertion order.
            elif slot:
                self.cpu_hits += 1
                output[i] = self.cpu[slot - 1]
            else:
                self.ssd_rows += 1
                # A miss uses the requested physical row, never the logical ID.
                output[i] = self.payload[row]
                if self.capacity_pages:
                    if len(self.pages) == self.capacity_pages:
                        self.pages.popitem(last=False)
                        self.evictions += 1
                    self.pages[page] = self.payload[page * 8:(page + 1) * 8].copy()
        return output

    def report(self):
        result = serving(self.requests, self.cpu_hits, self.gpu_hits, self.ssd_rows,
                         self.gpu_hits + self.ssd_rows)
        result.update(source='serial_cpu_oracle_only', native_execution=False, raw_ssd_access=False,
                      gpu_feature_cache_bytes=self.capacity_pages * 4096, cpu_feature_bytes=self.cpu.nbytes,
                      fifo_evictions=self.evictions, storage_request_sha256=self.address_digest.hexdigest(),
                      caveat='Serial FIFO semantics only; no native concurrency, latency, command coalescing or performance estimate')
        return result
