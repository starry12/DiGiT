"""Validated CPU trace templates; each iteration yields private batch storage."""
import time
from pathlib import Path

import dgl
import torch
from digit.eval_trace import EvaluationTrace, EvaluationTraceError

CACHE_BUDGET_BYTES = 2**30


def identity(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def estimate_trace(manifest):
    batches = []
    for rec in manifest['batches']:
        size = 8 * (rec['input_nodes'] + rec['output_nodes'])
        for layer in rec['layers']:
            size += 8 * (layer['num_src_nodes'] + layer['num_dst_nodes'])
            size += (24 if 'eids_sha256' in layer else 16) * layer['num_edges']
        batches.append(size)
    # CPU COO templates only. Reserve object overhead and a transient decoded
    # batch plus its copies; no persistent CUDA topology/feature allocation.
    overhead = sum(65536 + 4096 * len(r['layers']) for r in manifest['batches'])
    return dict(tensor_bytes=sum(batches), object_reserve_bytes=overhead,
                transient_reserve_bytes=2 * max(batches, default=0),
                required_bytes=sum(batches) + overhead + 2 * max(batches, default=0))


class PreparedEvaluationTrace(EvaluationTrace):
    def __init__(self, root, expected_sha256):
        super().__init__(root, verify_files=True)
        self._manifest_sha256 = super().manifest_sha256
        if self._manifest_sha256 != expected_sha256:
            raise EvaluationTraceError('Trace does not match the frozen input manifest')
        self.budget = estimate_trace(self.manifest)
        self._templates = None
        self._identities = None
        self.preparation = None

    @property
    def manifest_sha256(self):
        return self._manifest_sha256

    def _paths(self):
        return [self.manifest_path] + [self.root / r['path'] for r in self.manifest['batches']]

    def _check_identities(self, expected):
        if [identity(p) for p in self._paths()] != expected:
            raise EvaluationTraceError('Trace files changed; rebuild the verified cache')

    def prepare(self, max_bytes=CACHE_BUDGET_BYTES):
        if self._templates is not None:
            self._check_identities(self._identities)
            return self.preparation
        if self.budget['required_bytes'] > max_bytes:
            raise EvaluationTraceError('Trace cache exceeds its explicit host budget')
        before = [identity(p) for p in self._paths()]
        # Detect a manifest replacement between construction and preparation.
        if super().manifest_sha256 != self._manifest_sha256:
            raise EvaluationTraceError('Trace manifest changed before preparation')
        start = time.perf_counter()
        # The original path performs file hashes, semantic checks and block
        # construction. Do all of them once, on CPU, before admitting this cache.
        templates = list(super().iter_batches(device=None))
        self._check_identities(before)
        if len(templates) != self.manifest['sampling']['num_batches']:
            raise EvaluationTraceError('Incomplete prepared trace')
        self._templates = templates
        self._identities = before
        self.preparation = dict(seconds=time.perf_counter()-start,
            batches=len(templates), manifest_sha256=self.manifest_sha256,
            storage='CPU COO templates', persistent_cuda_bytes=0, **self.budget)
        return self.preparation

    def iter_batches(self, device=None):
        if self._templates is None:
            raise EvaluationTraceError('Explicit trace preparation is required before evaluation')
        self._check_identities(self._identities)
        destination = torch.device('cpu' if device is None else device)
        for inp, out, blocks in self._templates:
            fresh = []
            for block in blocks:
                # Clone CPU tensors/topology so CPU users cannot poison the
                # cache through inplace writes. CUDA copies are already private.
                if destination.type == 'cpu':
                    u, v = block.edges(order='eid')
                    b = dgl.create_block((u.clone(), v.clone()),
                        num_src_nodes=block.num_src_nodes(), num_dst_nodes=block.num_dst_nodes())
                    for key, value in block.srcdata.items(): b.srcdata[key] = value.clone()
                    for key, value in block.dstdata.items(): b.dstdata[key] = value.clone()
                    for key, value in block.edata.items(): b.edata[key] = value.clone()
                else:
                    b = block.to(destination)
                fresh.append(b)
            yield inp.to(destination, copy=True), out.to(destination, copy=True), fresh
        self._check_identities(self._identities)
