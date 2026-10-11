"""Connect owned native blocks to row-addressed features and SAGE/Adam.

No graph allocation, GPU registration, hot-set allocation or SSD operation is
performed here. An admitted caller supplies the native sampler and feature
provider. CPU tests use actual native CPU sampling over tiny anonymous arenas.
"""
from dataclasses import dataclass
import hashlib
import time
import numpy as np
import torch
import dgl
from ae.igb.models import SAGE
from candidates.ukl_native_sampling_v10r4 import sampling as S, fork_guard as H
from candidates.ukl_sage_compact_v1.sampler import STORAGE_ROW


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare_model(seed=0):
    """Run lazy CPU framework paths BEFORE large/pinned arena ownership."""
    if H._ACTIVE or torch.cuda.is_initialized():
        raise RuntimeError('Model preparation must precede arena ownership and CUDA')
    H.prewarm_cpu_block()
    state = torch.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        # Warm lazy Adam/backward code with disposable parameters. Actual model
        # below starts with empty Adam state and performs no hidden warmup update.
        scratch = SAGE(128, 128, 19, num_layers=3, dropout=.2)
        opt = torch.optim.Adam(scratch.parameters(), lr=.001, weight_decay=.001)
        sum(p.square().sum() for p in scratch.parameters()).backward(); opt.step()
        del opt, scratch
        torch.random.default_generator.manual_seed(seed)
        model = SAGE(128, 128, 19, num_layers=3, dropout=.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
        model.train()
        return model, optimizer
    finally:
        torch.set_rng_state(state)


@dataclass(frozen=True)
class FeatureRequest:
    arm: str
    logical_ids: np.ndarray
    storage_rows: np.ndarray
    row_count: int
    row_bytes: int = 512

    def byte_offsets(self, region_offset=0):
        limit = np.iinfo(np.int64).max
        if (self.storage_rows.dtype != np.int64 or self.storage_rows.ndim != 1
                or type(self.row_count) is not int or self.row_count <= 0
                or np.any(self.storage_rows < 0) or np.any(self.storage_rows >= self.row_count)
                or type(region_offset) is not int or region_offset < 0 or region_offset % 512
                or region_offset > limit - self.row_count * 512):
            raise ValueError('Region alignment or signed 64-bit address overflow')
        return self.storage_rows * np.int64(512) + np.int64(region_offset)


def request_from_blocks(arm, graph, inputs, targets, blocks, roots):
    if arm not in ('gids', 'digit') or len(blocks) != 3:
        raise ValueError('Expected arm and three SAGE blocks')
    if inputs.device.type != 'cpu' or targets.device.type != 'cpu':
        raise ValueError('Owned sampler currently returns CPU block metadata')
    if not np.array_equal(targets.numpy(), roots):
        raise ValueError('Root order changed')
    for i, b in enumerate(blocks):
        src, dst = b.srcdata[dgl.NID], b.dstdata[dgl.NID]
        if not torch.equal(src[:len(dst)], dst):
            raise ValueError('SAGE destination prefix mismatch')
        if i and not torch.equal(blocks[i-1].dstdata[dgl.NID], src):
            raise ValueError('Block chain mismatch')
    if (not torch.equal(inputs, blocks[0].srcdata[dgl.NID])
            or not torch.equal(targets, blocks[-1].dstdata[dgl.NID])):
        raise ValueError('Block endpoints differ from sampled IDs')
    ids = inputs.numpy().astype(np.int64, copy=True)
    row_tensor = blocks[0].srcdata[STORAGE_ROW]
    if row_tensor.dtype != torch.int64 or row_tensor.device.type != 'cpu':
        raise ValueError('Storage addresses must remain int64 CPU metadata')
    rows = row_tensor.numpy().copy()
    bound = graph.nodes if arm == 'gids' else graph.storage_rows
    if (rows.shape != ids.shape or np.any(ids < 0) or np.any(ids >= graph.nodes)
            or len(np.unique(ids)) != len(ids) or np.any(rows < 0) or np.any(rows >= bound)):
        raise ValueError('Feature IDs or addresses outside declared layout')
    if arm == 'gids' and not np.array_equal(ids, rows):
        raise ValueError('GIDS must address original feature rows')
    ids.setflags(write=False); rows.setflags(write=False)
    return FeatureRequest(arm, ids, rows, bound)


class Trainer:
    """One arm, one optimizer; provider(request) returns ordered float32 features.

    Provider owns cache/SSD policy and lifetime: hot lookup uses logical_ids,
    device reads use storage_rows. No fabricated I/O or cache-hit counters.
    Device must match the supplied already initialized model; this class does
    not select GPUs or register graph memory. Explicit synchronize is used only
    around the measured window; step diagnostics are not performance timings.
    """
    def __init__(self, native, arm, model, optimizer, provider, seed=0,
                 check=lambda: None, synchronize=lambda: None):
        if type(native) is not S.Native or arm not in ('gids', 'digit'):
            raise ValueError('Exact owned native sampler and valid arm required')
        if native.graph.nodes > 4096:
            raise RuntimeError('CPU training adapter is bounded to small fixtures')
        H.assert_dontfork(native.graph.arena)
        if not H._ACTIVE:
            raise RuntimeError('Anonymous ownership guard must remain active')
        if {id(p) for g in optimizer.param_groups for p in g['params']} != {id(p) for p in model.parameters()}:
            raise ValueError('Optimizer belongs to another model')
        self.native, self.arm = native, arm
        self.model, self.optimizer, self.provider = model, optimizer, provider
        self.sampler = S.Sampler(native, grouped=arm == 'digit', seed=seed)
        self.check, self.synchronize = check, synchronize
        self.device = next(model.parameters()).device
        if self.device.type != 'cpu':
            raise RuntimeError('GPU training stays closed until SSD binding and admitted worker are implemented')
        self.updates = 0
        self._rng = torch.Generator(device='cpu').manual_seed(seed).get_state()

    def step(self, roots, labels, batch, *, observe=None):
        self.check()
        roots, labels = np.asarray(roots), np.asarray(labels)
        if (type(batch) is not int or batch != self.updates
                or roots.ndim != 1 or labels.shape != roots.shape or labels.dtype != np.int64
                or labels.size == 0 or np.any(labels < 0) or np.any(labels >= 19)):
            raise ValueError('Labels must align with this root batch and be in [0,19)')
        inp, out, blocks = self.sampler.sample_blocks(roots, batch)
        request = request_from_blocks(self.arm, self.native.graph, inp, out, blocks, roots)
        self.check()
        x = self.provider(request)
        if (not isinstance(x, torch.Tensor) or x.shape != (len(inp), 128)
                or x.dtype != torch.float32 or x.device != self.device or x.requires_grad
                or not torch.isfinite(x).all().item()):
            raise ValueError('Provider must return finite ordered float32 features on model device')
        target = torch.from_numpy(labels.copy()).to(self.device)
        self.check()
        self.model.train()
        outer_rng = torch.get_rng_state()
        torch.set_rng_state(self._rng)
        try:
            pred = self.model(blocks, x)
            self._rng = torch.get_rng_state()
        finally:
            torch.set_rng_state(outer_rng)
        loss = torch.nn.functional.cross_entropy(pred, target)
        if not torch.isfinite(loss).item() or not torch.isfinite(pred).all().item():
            raise RuntimeError('Nonfinite forward/loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in self.model.parameters()):
            raise RuntimeError('Missing or nonfinite gradients')
        self.check()
        if observe is not None:
            observe(request, blocks, x, target, pred, loss, self.model)
        self.optimizer.step()
        if any(not torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise RuntimeError('Nonfinite updated parameters')
        self.updates += 1
        return dict(batch=batch, roots=len(roots), input_rows=len(inp),
                    edges=[b.num_edges() for b in blocks], loss=float(loss.detach()))

    def window(self, roots, labels, *, warmup=20, measured=300, fixture=False):
        roots, labels = np.asarray(roots), np.asarray(labels)
        if (type(warmup) is not int or type(measured) is not int or warmup < 0 or measured <= 0
                or roots.ndim != 2 or roots.shape != labels.shape or roots.dtype != np.int64
                or labels.dtype != np.int64 or len(roots) != warmup + measured
                or len(np.unique(roots)) != roots.size):
            raise ValueError('Complete distinct root window required')
        if fixture:
            if self.native.graph.nodes > 4096 or roots.size > 4096 or len(roots) > 16:
                raise ValueError('CPU fixture bound')
        else:
            # Full-graph resource admission and SSD integration are next tasks.
            raise RuntimeError('Full UKL training is not enabled by this CPU adapter')
        if self.updates:
            raise RuntimeError('Window requires a fresh trainer')
        initial = model_hash(self.model); rows = []; start = None
        for i in range(len(roots)):
            if i == warmup:
                self.synchronize(); start = time.perf_counter()
            rows.append(self.step(roots[i], labels[i], i))
        self.synchronize(); elapsed = time.perf_counter() - start
        if self.updates != warmup + measured or model_hash(self.model) == initial:
            raise RuntimeError('Training update coverage failed')
        return dict(passed=True, arm=self.arm, updates=self.updates, warmup_updates=warmup,
                    measured_updates=measured, measured_seconds=elapsed,
                    timing_scope='CPU fixture only; not native GPU/SSD performance',
                    batches=rows, accuracy=False, evaluation_calls=0,
                    gpu_called=False, raw_ssd_access=False, initial_model_sha256=initial,
                    final_model_sha256=model_hash(self.model))
