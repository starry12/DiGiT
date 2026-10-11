"""Three scheduling modes with real CUDA sampling/model and simulated feature I/O.

The fake store implements the actual loader's raw-pointer API on CUDA streams.
It tests Python scheduling, pointer lifetime, stream dependencies and completion
handling; it does not validate the native BaM ticket implementation or SSD I/O.
All graph/feature/loss audits run after the 33 model updates, outside that loop.
"""
import hashlib
import random
import subprocess
import tempfile

from .common import *


class _CudaArray:
    """Borrow a loader-owned CUDA allocation without accessing it on the CPU."""
    def __init__(self, pointer, shape, typestr):
        self.__cuda_array_interface__ = dict(
            shape=tuple(shape), strides=None, typestr=typestr,
            data=(int(pointer), False), version=3,
        )


class FakeAsyncStore:
    """CUDA gathers with event-backed tickets and per-read CPU-count receipts."""
    def __init__(self, store):
        import torch
        self.torch = torch
        self.store = store
        self.device = store.device
        self.stream_pointer = None
        self.hints = []
        self.reads = []
        self.events = []
        self.read_groups = []
        self.cpu_counts = []
        self.legacy_cpu = 0
        self.live_ticket = None
        self.submitted = self.collected = self.max_outstanding = 0

    def _view(self, pointer, shape, typestr):
        # No producer stream is advertised: the loader explicitly establishes
        # dependencies before calling this API, just as with the native backend.
        value = self.torch.as_tensor(_CudaArray(pointer, shape, typestr), device=self.device)
        require(value.data_ptr() == pointer, 'Fake pointer adapter unexpectedly copied storage')
        return value

    def _stream(self, pointer):
        require(pointer == self.stream_pointer, 'Fake backend stream changed')
        return self.torch.cuda.ExternalStream(pointer, device=self.device)

    def prepare_async_reads(self, stream_pointer):
        require(self.stream_pointer is None, 'Fake async store was prepared twice')
        self.stream_pointer = stream_pointer

    def _hint(self, pointer, size, offset):
        require(offset == 0, 'Unexpected homogeneous feature offset')
        rows = self._view(pointer, (size,), '<i8')
        self.events.append(('hint', len(self.hints)))
        self.hints.append(rows.clone())

    def set_window_buffering(self, pointer, size, offset):
        self._hint(pointer, size, offset)

    def set_window_buffering_async(self, pointer, size, offset, stream_pointer):
        with self.torch.cuda.stream(self._stream(stream_pointer)):
            self._hint(pointer, size, offset)

    def _gather(self, count, outputs, indices, sizes, dim, cache_dim):
        require(count == len(outputs) == len(indices) == len(sizes), 'Invalid fake read count')
        require(dim == cache_dim == self.store.shape[1], 'Invalid fake feature width')
        group = list(range(len(self.reads), len(self.reads) + count))
        self.events.append(('read', tuple(group)))
        self.read_groups.append(group)
        refs = []
        for pointer, index, size in zip(outputs, indices, sizes):
            rows = self._view(index, (size,), '<i8')
            output = self._view(pointer, (size, dim), '<f4')
            self.reads.append(rows.clone())
            output.copy_(self.store.index_select(0, rows))
            refs.extend((rows, output))
        # Deliberately all hits, exercising the accumulator's high-hit cap.
        cpu_count = sum(sizes)
        self.cpu_counts.append(cpu_count)
        return refs, cpu_count

    def read_feature_merged(self, count, outputs, indices, sizes, dim, cache_dim):
        refs, self.legacy_cpu = self._gather(count, outputs, indices, sizes, dim, cache_dim)
        self.torch.cuda.current_stream(self.device).synchronize()
        del refs

    def get_cpu_access_count(self):
        return self.legacy_cpu

    def flush_cpu_access_count(self):
        self.legacy_cpu = 0

    def read_feature_merged_async(self, count, outputs, indices, sizes, dim, cache_dim,
                                  stream_pointer):
        require(self.live_ticket is None, 'Two fake reads were outstanding')
        stream = self._stream(stream_pointer)
        with self.torch.cuda.stream(stream):
            refs, cpu_count = self._gather(count, outputs, indices, sizes, dim, cache_dim)
            event = self.torch.cuda.Event()
            event.record(stream)
        ticket = dict(event=event, references=refs, cpu_count=cpu_count,
                      number=self.submitted, collected=False)
        self.live_ticket = ticket
        self.submitted += 1
        self.max_outstanding = max(self.max_outstanding, 1)
        return ticket

    def collect_feature_read(self, ticket):
        require(ticket is self.live_ticket and not ticket['collected'], 'Invalid fake read ticket')
        ticket['event'].synchronize()
        ticket['collected'] = True
        ticket['references'] = []
        self.live_ticket = None
        self.collected += 1
        return ticket['cpu_count']


class _FixtureDataLoader:
    """Expose the real scheduler's dataloader_it contract over fixed tiny roots."""
    def __init__(self, source, loader):
        self.dataloader_it = iter(source)
        self.loader = loader

    def __iter__(self):
        return self

    def __next__(self):
        return self.loader.fetch_feature(128, self.dataloader_it, 'cuda:0')


def _tensor_digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _tree_digest(value):
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(_tensor_digest([item]).encode())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _compare_trees(expected, actual):
    import torch
    result = dict(passed=True, bit_exact=True, max_abs_error=0., tensor_comparisons=0,
                  rtol=1e-5, atol=1e-6)
    def visit(left, right):
        if torch.is_tensor(left):
            require(torch.is_tensor(right), 'State tensor type changed')
            torch.testing.assert_close(left, right, rtol=result['rtol'], atol=result['atol'])
            result['bit_exact'] = result['bit_exact'] and torch.equal(left, right)
            result['tensor_comparisons'] += 1
            if left.numel():
                result['max_abs_error'] = max(result['max_abs_error'],
                                               float((left - right).abs().max().item()))
        elif isinstance(left, dict):
            require(isinstance(right, dict) and left.keys() == right.keys(), 'State keys changed')
            for key in left:
                visit(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            require(type(left) is type(right) and len(left) == len(right), 'State sequence changed')
            for a, b in zip(left, right):
                visit(a, b)
        else:
            require(left == right, 'Optimizer/state scalar changed')
    visit(expected, actual)
    return result


def exercise(graph, arrays, bundle, features, mode):
    import numpy as np
    import torch
    import dgl
    from candidates.pa_sage_512b_latency_v1.sampler import KnownExtentSampler
    from candidates.pa_sage_512b_latency_v1.model import FunctionalMeanSAGE
    from candidates.pa_sage_512b_pair_v4.loader import BoundedGIDS
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks
    from .loader import OverlapGIDS, batches as scheduled_batches

    require(mode in VARIANTS, 'Unknown tiny scheduling mode')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dgl.seed(0)
    dgl.random.seed(0)
    sampler = KnownExtentSampler([4, 3, 2], bundle, cuda_mode='required',
                                 metadata_mode='gpu_i32_uva_eid64', random_seed=0)
    sampler._ensure_cuda_metadata(graph, torch.device('cuda:0'))
    store = torch.tensor(np.asarray(bundle.arrays['reordered_features']), device='cuda:0')
    native = FakeAsyncStore(store)
    klass = BoundedGIDS if mode == 'legacy' else OverlapGIDS
    loader = klass.__new__(klass)
    loader.accumulator_flag = loader.window_buffering_flag = True
    loader.heterograph = False
    loader.wb_size = 2
    loader.required_accesses = 6832.4
    loader.cache_dim = 128
    loader.gids_device = 'cuda:0'
    loader.feature_index_mode = 'explicit'
    loader.graph_reorganize = False
    loader.feature_map = None
    loader.strict_feature_row_validation = True
    loader.feature_index_batches = {key: 0 for key in ('logical', 'explicit', 'legacy_map')}
    loader.feature_index_rows = dict(loader.feature_index_batches)
    loader.BAM_FS = native
    loader.begin_epoch()

    rng_prepare_checks = []
    rng_sample_checks = []
    def checked_rng_call(callback, records):
        before = (torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0'))
        result = callback()
        after = (torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0'))
        require(all(torch.equal(a, b) for a, b in zip(before, after)),
                'Sampling/preparation consumed the model Torch RNG')
        records.append(_tensor_digest(before))
        return result
    if mode != 'legacy':
        prepare = loader.prepare_next
        loader.prepare_next = lambda: checked_rng_call(prepare, rng_prepare_checks)

    roots = [torch.tensor([0, 2, 4, 6], device='cuda:0') for _ in range(32)]
    roots.append(torch.tensor([8], device='cuda:0'))
    def source():
        for root in roots:
            yield checked_rng_call(lambda: sampler.sample(graph, root), rng_sample_checks)
    dataloader = _FixtureDataLoader(source(), loader)
    model = FunctionalMeanSAGE(128, 128, 172, 3, .2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    initial_state = _cpu_tree(model.state_dict())
    audit_batches = []
    losses = []
    model_rng_states = []
    try:
        for inp, out, blocks, x in scheduled_batches(dataloader, loader, mode):
            # Retain tiny graph/tensor owners, then audit them after all updates.
            # No per-batch .item(), feature-to-CPU copy or graph verification.
            audit_batches.append((inp, out, blocks, x.detach()))
            model_rng_states.append((torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0')))
            loss = torch.nn.functional.cross_entropy(model([block.int() for block in blocks], x), out % 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        torch.cuda.synchronize()
    finally:
        if mode != 'legacy':
            loader.drain()

    signatures, feature_signatures, seen = [], [], []
    for inp, out, blocks, x in audit_batches:
        verify_digit_blocks(blocks, arrays, [4, 3, 2])
        ids = inp.cpu().numpy()
        rows = blocks[0].srcdata['digit_storage_row']
        np.testing.assert_array_equal(bundle.arrays['storage_to_node'][rows.cpu().numpy()], ids)
        np.testing.assert_array_equal(x.cpu().numpy().view('u4'), features[ids].view('u4'))
        tensors = [inp, out]
        for block in blocks:
            tensors.extend((block.srcdata[dgl.NID], block.dstdata[dgl.NID],
                            block.edata[dgl.EID], *block.edges(order='eid')))
        tensors.extend((rows, blocks[0].srcdata['digit_storage_is_group'],
                        blocks[0].dstdata['digit_sampled_groups'],
                        blocks[0].dstdata['digit_sampled_nodes']))
        signatures.append(_tensor_digest(tensors))
        feature_signatures.append(_tensor_digest([x]))
        seen.extend(out.cpu().tolist())
    require(seen == [0, 2, 4, 6] * 32 + [8], 'Lost root or tail in tiny pipeline')
    require(len(losses) == len(signatures) == len(rng_sample_checks) == 33, 'Wrong update count')
    loss_values = torch.stack(losses).cpu()
    require(bool(torch.isfinite(loss_values).all()), 'Nonfinite tiny pipeline loss')
    require(len(native.hints) == len(native.reads) == 33, 'Missing tiny feature hints/reads')
    for hinted, read_rows in zip(native.hints, native.reads):
        require(torch.equal(hinted, read_rows), 'Hint/read physical addresses differ')
    pipe = loader.pipeline.summary()
    require(pipe['drained'] and pipe['delivered_batches'] == pipe['read_batches'] == 33,
            'Tiny pipeline failed to drain')
    require(pipe['max_merged_batches'] <= 4 and pipe['max_pending_batches'] <= 5,
            'Tiny pipeline retained too many sampled batches')
    require(pipe['max_read_rows'] <= 262144, 'Tiny merged read exceeded row cap')
    require(loader.delivered_rows == sum(len(batch[0]) for batch in audit_batches), 'Lost feature rows')
    require(loader.feature_index_batches['explicit'] == 66, 'Extra/missing feature-row resolution')
    async_summary = None
    if mode != 'legacy':
        require(len(rng_prepare_checks) == 34, 'Wrong preparation count including EOF')
        require(pipe['max_outstanding_reads'] == 1 and pipe['outstanding_reads'] == 0,
                'Python ticket did not drain')
        require(pipe['submitted_reads'] == pipe['collected_reads'] == pipe['read_calls'],
                'Python ticket counters differ')
        require(native.live_ticket is None and native.submitted == native.collected == pipe['read_calls'],
                'Fake CUDA ticket did not drain')
        require(not loader.hint_refs and loader.live_ticket is None, 'Loader kept active I/O references')
        async_summary = loader.async_summary()

    state = dict(model=_cpu_tree(model.state_dict()), adam=_cpu_tree(optimizer.state_dict()),
                 losses=loss_values, initial_model=initial_state)
    result = dict(
        mode=mode, variant=mode, model_updates=33, roots=129, tail=1,
        signatures=signatures, feature_signatures=feature_signatures,
        losses=loss_values.tolist(), pipeline=pipe, async_backend=async_summary,
        hint_signatures=[_tensor_digest([rows]) for rows in native.hints],
        read_signatures=[_tensor_digest([rows]) for rows in native.reads],
        hint_read_events=native.events, read_groups=native.read_groups, cpu_counts=native.cpu_counts,
        prepare_rng_unchanged=True, prepare_rng_checks=len(rng_prepare_checks),
        sampling_rng_unchanged=True, sampling_rng_checks=len(rng_sample_checks),
        model_rng_signatures=[_tensor_digest(states) for states in model_rng_states],
        final_rng_sha256=_tensor_digest([torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0')]),
        initial_model_sha256=_tree_digest(initial_state),
        final_model_sha256=_tree_digest(state['model']), final_adam_sha256=_tree_digest(state['adam']),
        feature_index=loader.get_feature_index_stats(),
        feature_reads_simulated=True, native_ticket_implementation_tested=False,
        real_cuda_stream_gather=True, post_training_audits=True,
    )
    return result, state


def main():
    require(not (OUT / 'gpu_checks.json').exists(), 'Preserve GPU evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'Use physical GPU2')
    fields = subprocess.check_output([
        'nvidia-smi', '-i', '2', '--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    require(fields[0].strip() == GPU_UUID and int(fields[1]) < 1024, 'GPU busy or changed')
    tested_names = {'common.py', 'loader.py', 'window.py', 'gpu_tests.py'}
    tested = {str(path.relative_to(ROOT)): sha(path) for path in source_files()
              if path.name in tested_names}
    setup()
    reference_sha256 = reference.verify()
    import torch
    from candidates.pa_sage_512b_pair_v1.tests import fixture
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    comparisons = {}
    baseline_result = baseline_state = None
    with tempfile.TemporaryDirectory(prefix='digit-overlap-') as folder:
        graph, arrays, bundle, features = fixture(folder, pin=True)
        try:
            for mode in VARIANTS:
                result, state = exercise(graph, arrays, bundle, features, mode)
                if baseline_result is None:
                    baseline_result, baseline_state = result, state
                else:
                    for key in ('signatures', 'feature_signatures', 'hint_signatures',
                                'read_signatures', 'hint_read_events', 'read_groups', 'cpu_counts',
                                'model_rng_signatures', 'initial_model_sha256', 'final_rng_sha256'):
                        require(result[key] == baseline_result[key], 'Tiny parity changed ' + key + ': ' + mode)
                    comparisons[mode] = _compare_trees(baseline_state, state)
                results.append(result)
                torch.cuda.empty_cache()
        finally:
            graph._graph.unpin_memory_()
    final_tested = {str(path.relative_to(ROOT)): sha(path) for path in source_files()
                    if path.name in tested_names}
    changed = sorted(key for key in set(tested) | set(final_tested)
                     if tested.get(key) != final_tested.get(key))
    require(not changed, 'Overlap source changed during GPU checks: ' + ', '.join(changed))
    write(OUT / 'gpu_checks.json', dict(
        passed=True, results=results, comparisons=comparisons, model_updates=99,
        total_model_updates=99, exact_block_and_row_parity=True, exact_feature_parity=True,
        exact_model_rng_parity=True, model_and_adam_numerical_parity=True,
        model_and_adam_bit_exact=all(item['bit_exact'] for item in comparisons.values()),
        feature_reads_simulated=True, native_ticket_implementation_tested=False,
        raw_ssd_access=False, performance_evidence=False,
        fixture_dataloader_adapter=True, actual_loader_scheduler=True,
        binary_sha256=sha(BINARY), reference_sha256=reference_sha256, tested_sha256=tested,
    ))
    print('GPU legacy/serial/overlap parity passed: 99 updates; simulated feature I/O, no SSD', flush=True)


if __name__ == '__main__':
    main()
