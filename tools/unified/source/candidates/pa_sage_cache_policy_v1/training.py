"""Shared sampling/profile/one-epoch model adapter; CPU fixtures use real DGL."""
import hashlib
import importlib.util
import math
import random
import time
import numpy as np
from .common import ROOT, require, digest
from .selection import FrequencyProfile


def seed_cpu(value):
    import torch
    import dgl
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    dgl.seed(value)


def model_hash(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def create_cpu_model(p):
    import torch
    seed_cpu(p['seed'])
    spec = importlib.util.spec_from_file_location('_cache_policy_models', ROOT / 'ae/papers/runtime/models.py')
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    model = models.SAGE(128, p['hidden'], p['classes'], num_layers=p['layers'], dropout=p['dropout'])
    kwargs = dict(p['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    initial = model_hash(model)
    seed_cpu(p['seed'])
    model.train()
    return model, optimizer, initial


def trace_update(h, inp, out, blocks):
    import dgl
    from digit.sampler import DIGIT_STORAGE_ROW, DIGIT_STORAGE_IS_GROUP
    tensors = [inp, out, blocks[0].srcdata[DIGIT_STORAGE_ROW], blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP]]
    for block in blocks:
        u, v = block.edges(order='eid')
        tensors.extend((block.srcdata[dgl.NID], block.dstdata[dgl.NID], u, v))
    for tensor in tensors:
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())


def profile_cpu(sampler, graph, train_ids, nodes, seed, batches, batch_size):
    import torch
    require(nodes <= 4096 and graph.num_edges() <= 131072, 'CPU profile is limited to small fixtures')
    require(len(train_ids) > 0 and batch_size > 0 and 0 < batches <= math.ceil(len(train_ids) / batch_size), 'Invalid profile extent')
    seed_cpu(seed)
    order = np.random.default_rng(seed).permutation(np.asarray(train_ids, dtype=np.int64))
    frequency = FrequencyProfile(nodes, seed)
    trace = hashlib.sha256()
    start = time.perf_counter()
    for i in range(batches):
        roots = torch.from_numpy(order[i * batch_size:(i + 1) * batch_size].copy())
        inp, out, blocks = sampler.sample_blocks(graph, roots)
        frequency.observe(inp.numpy().astype(np.int64, copy=False))
        trace_update(trace, inp, out, blocks)
    result = frequency.freeze()
    result.update(sampling_trace_sha256=trace.hexdigest(), root_sha256=digest(order[:batches * batch_size]),
                  preparation_seconds=time.perf_counter() - start, fixture=True, native_execution=False)
    return frequency.counts, result


def cpu_epoch(p, sampler, graph, train_ids, labels, oracle):
    """A complete tiny epoch, with actual forward/backward/Adam and no evaluation."""
    import torch
    from digit.sampler import DIGIT_STORAGE_ROW
    require(p['epochs'] == 1 and p['evaluation'] == 'disabled', 'Only the one-epoch performance protocol')
    require(graph.num_nodes() <= 4096 and graph.num_edges() <= 131072, 'CPU training is limited to small fixtures')
    model, optimizer, initial = create_cpu_model(p)
    roots = np.random.default_rng(np.random.SeedSequence([p['seed'], 0])).permutation(train_ids)
    trace = hashlib.sha256()
    losses, shapes = [], []
    examples = 0
    for lo in range(0, len(roots), p['batch_size']):
        targets = torch.from_numpy(roots[lo:lo + p['batch_size']].copy())
        inp, out, blocks = sampler.sample_blocks(graph, targets)
        require(torch.equal(out, targets), 'Sampler changed output root order')
        rows = blocks[0].srcdata[DIGIT_STORAGE_ROW].numpy().astype(np.int64, copy=False)
        logical = inp.numpy().astype(np.int64, copy=False)
        x = torch.from_numpy(oracle.fetch(logical, rows))
        pred = model(blocks, x)
        loss = torch.nn.functional.cross_entropy(pred, torch.from_numpy(labels[out.numpy()].copy()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        require(torch.isfinite(pred).all().item() and torch.isfinite(loss).item() and
                all(param.grad is None or torch.isfinite(param.grad).all().item() for param in model.parameters()),
                'Nonfinite model computation')
        optimizer.step()
        losses.append(float(loss.detach()))
        examples += len(out)
        shapes.append(dict(input_nodes=len(inp), output_nodes=len(out), block_edges=[b.num_edges() for b in blocks]))
        trace_update(trace, inp, out, blocks)
    require(len(losses) == math.ceil(len(roots) / p['batch_size']) and examples == len(roots), 'Incomplete fixture epoch')
    final = model_hash(model)
    require(final != initial and all(torch.isfinite(param).all().item() for param in model.parameters()), 'Model did not update correctly')
    return dict(passed=True, source='cpu_fixture', native_execution=False, epochs=1,
                updates=len(losses), examples=examples, initial_parameters_sha256=initial, final_parameters_sha256=final,
                root_sha256=digest(roots), sampling_trace_sha256=trace.hexdigest(), losses=losses,
                shapes=shapes, evaluation_calls=0, metrics=oracle.report(),
                training_seconds=None, note='CPU correctness only; no native time or hit-rate claim')
