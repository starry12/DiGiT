"""Real CUDA incremental-group/training parity with the frozen overlap scheduler and fake I/O.

Each fanout case runs 33 updates for the original sampler, the warmed eager model and its dense CUDA Graph model. Feature reads are CUDA gathers through
the accepted FakeAsyncStore, not BaM reads or an SSD acceptance experiment.
"""
import random
import subprocess
import tempfile

from .common import *
from candidates.pa_sage_512b_overlap_v1.gpu_tests import (
    FakeAsyncStore, _FixtureDataLoader, _tensor_digest, _cpu_tree, _tree_digest,
    _compare_trees,
)


MODEL_VARIANTS = ('original', 'legacy', 'incremental')
FANOUT_CASES = ((4, 3, 2), (10, 5, 5))
TESTED_FILES = ('common.py', 'sampler.py', 'gpu_tests.py', 'trace.py')


def tested_sources():
    return {str((HERE / name).relative_to(ROOT)): sha(HERE / name)
            for name in TESTED_FILES}


def exercise(graph, arrays, bundle, features, variant, fanouts):
    import numpy as np
    import torch
    import dgl
    from candidates.pa_sage_512b_latency_v1.sampler import KnownExtentSampler
    from candidates.pa_sage_512b_latency_v1.model import FunctionalMeanSAGE
    from candidates.pa_sage_512b_overlap_v1.loader import OverlapGIDS, batches
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks
    from candidates.pa_sage_dense_graph_v1.model import DenseGraphSAGE, prepare_model
    from .sampler import IncrementalGroupSampler

    require(variant in MODEL_VARIANTS, 'Unknown tiny group variant')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dgl.seed(0)
    dgl.random.seed(0)
    kwargs = dict(cuda_mode='required', metadata_mode='gpu_i32_uva_eid64', random_seed=0)
    sampler = (KnownExtentSampler(list(fanouts), bundle, **kwargs) if variant=='original' else
               IncrementalGroupSampler(list(fanouts),bundle,group_mode=variant,**kwargs))
    sampler._ensure_cuda_metadata(graph, torch.device('cuda:0'))
    store = torch.tensor(np.asarray(bundle.arrays['reordered_features']), device='cuda:0')
    fake = FakeAsyncStore(store)
    loader = OverlapGIDS.__new__(OverlapGIDS)
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
    loader.BAM_FS = fake
    loader.begin_epoch()

    prepare_rng, sampling_rng = [], []
    def checked_rng_call(callback, records):
        before = (torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0'))
        result = callback()
        after = (torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0'))
        require(all(torch.equal(a, b) for a, b in zip(before, after)),
                'Sampling/preparation consumed the model Torch RNG')
        records.append(_tensor_digest(before))
        return result

    prepare = loader.prepare_next
    loader.prepare_next = lambda: checked_rng_call(prepare, prepare_rng)
    roots = [torch.tensor([0, 2, 4, 6], device='cuda:0') for _ in range(32)]
    roots.append(torch.tensor([8], device='cuda:0'))
    def source():
        for root in roots:
            yield checked_rng_call(lambda: sampler.sample(graph, root), sampling_rng)

    dataloader = _FixtureDataLoader(source(), loader)
    model = DenseGraphSAGE(128, 128, 172, 3, .2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    prepare_model(model, optimizer, 'graph', batch_size=4)
    initial_state = _cpu_tree(model.state_dict())
    audits, losses, logits_values, model_rng = [], [], [], []
    profiler = None
    trace_path = OUT/('tiny_'+variant+'_trace.json')
    if variant!='original' and list(fanouts)==[10,5,5]:
        require(not trace_path.exists(),'Preserve tiny trace')
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2,warmup=1,active=16,repeat=1),
            on_trace_ready=lambda p:p.export_chrome_trace(str(trace_path)),
            record_shapes=False,profile_memory=False,with_stack=False)
        profiler.__enter__()
    try:
        for inp, out, blocks, x in batches(dataloader, loader, 'overlap'):
            # Preserve only tiny owners; all graph/feature audits follow training.
            audits.append((inp, out, blocks, x.detach()))
            model_rng.append((torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0')))
            logits = model([block.int() for block in blocks], x)
            loss = torch.nn.functional.cross_entropy(logits, out % 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logits_values.append(logits.detach().clone())
            losses.append(loss.detach())
            if profiler is not None:profiler.step()
        torch.cuda.synchronize()
    finally:
        loader.drain()
        if profiler is not None:profiler.__exit__(None,None,None)

    block_signatures, feature_signatures, seen = [], [], []
    for inp, out, blocks, x in audits:
        verify_digit_blocks(blocks, arrays, fanouts)
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
        block_signatures.append(_tensor_digest(tensors))
        feature_signatures.append(_tensor_digest([x]))
        seen.extend(out.cpu().tolist())
    require(seen == [0, 2, 4, 6] * 32 + [8], 'Lost root or tail in tiny pipeline')
    require(len(losses) == len(block_signatures) == len(sampling_rng) == 33, 'Wrong update count')
    require(sampler._cuda_call_counter == 33, 'Wrong grouped sampling invocation count')
    loss_values = torch.stack(losses).cpu()
    require(bool(torch.isfinite(loss_values).all()), 'Nonfinite tiny pipeline loss')
    require(len(fake.hints) == len(fake.reads) == 33, 'Missing feature hints or reads')
    for hinted, read_rows in zip(fake.hints, fake.reads):
        require(torch.equal(hinted, read_rows), 'Hint/read physical addresses differ')
    pipe = loader.pipeline.summary()
    require(pipe['drained'] and pipe['sampled_batches'] == pipe['delivered_batches'] ==
            pipe['read_batches'] == 33, 'Tiny pipeline failed to drain')
    require(pipe['max_merged_batches'] <= 4 and pipe['max_pending_batches'] <= 5 and
            pipe['max_read_rows'] <= 262144, 'Tiny pipeline exceeded frozen bounds')
    require(loader.delivered_rows == sum(len(batch[0]) for batch in audits), 'Lost feature rows')
    require(loader.feature_index_batches['explicit'] == 66, 'Extra or missing row resolution')
    require(len(prepare_rng) == 34, 'Wrong preparation count including EOF')
    require(pipe['max_outstanding_reads'] == 1 and pipe['outstanding_reads'] == 0,
            'Python ticket did not drain')
    require(pipe['submitted_reads'] == pipe['collected_reads'] == pipe['read_calls'],
            'Python ticket counters differ')
    require(fake.live_ticket is None and fake.submitted == fake.collected == pipe['read_calls'],
            'Fake CUDA ticket did not drain')
    require(not loader.hint_refs and loader.live_ticket is None, 'Active I/O references retained')

    state = dict(initial_model=initial_state, model=_cpu_tree(model.state_dict()),
                 adam=_cpu_tree(optimizer.state_dict()), losses=loss_values,
                 logits=_cpu_tree(logits_values),
                 final_gradients=_cpu_tree([parameter.grad for parameter in model.parameters()]))
    result = dict(
        variant=variant, dense_graph=model.dense_summary(), io_mode='overlap', fanouts=list(fanouts),
        model_updates=33, roots=129, tail=1, native_grouped_batches=sampler._cuda_call_counter,
        block_signatures=block_signatures, feature_signatures=feature_signatures,
        losses=loss_values.tolist(), pipeline=pipe, async_backend=loader.async_summary(),
        hint_signatures=[_tensor_digest([rows]) for rows in fake.hints],
        read_signatures=[_tensor_digest([rows]) for rows in fake.reads],
        hint_read_events=fake.events, read_groups=fake.read_groups, cpu_counts=fake.cpu_counts,
        prepare_rng_unchanged=True, prepare_rng_checks=len(prepare_rng),
        sampling_rng_unchanged=True, sampling_rng_checks=len(sampling_rng),
        prepare_rng_signatures=prepare_rng, sampling_rng_signatures=sampling_rng,
        model_rng_signatures=[_tensor_digest(values) for values in model_rng],
        final_rng_sha256=_tensor_digest([torch.get_rng_state(), torch.cuda.get_rng_state('cuda:0')]),
        initial_model_sha256=_tree_digest(initial_state),
        final_model_sha256=_tree_digest(state['model']), final_adam_sha256=_tree_digest(state['adam']),
        logits_sha256=_tree_digest(state['logits']), final_gradients_sha256=_tree_digest(state['final_gradients']),
        feature_index=loader.get_feature_index_stats(), feature_reads_simulated=True,
        native_ticket_implementation_tested=False, real_cuda_stream_gather=True,
        post_training_audits=True, optimizer=dict(name='Adam', lr=.001,
            origin='Frozen overlap GPU training-parity fixture'),
    )
    return result, state


def main():
    require(not (OUT / 'gpu_checks.json').exists(), 'Preserve GPU evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'Use physical GPU2')
    fields = subprocess.check_output([
        'nvidia-smi', '-i', '2', '--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    require(fields[0].strip() == GPU_UUID and int(fields[1]) < 1024, 'GPU busy or changed')
    tested = tested_sources()
    binary_sha256 = sha(BINARY); sampler_binary_sha256=sha(SAMPLER_BINARY)
    setup()
    reference_sha256 = reference.verify()
    import torch
    import dgl
    import BAM_Feature_Store
    import DiGiTGroupIncrementalCUDA
    require(Path(DiGiTGroupIncrementalCUDA.__file__).resolve()==SAMPLER_BINARY, "Wrong group module")
    from candidates.pa_sage_512b_pair_v1.tests import fixture
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve() == BINARY,
            'Wrong frozen BaM binary imported')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results, comparisons = [], {}
    parity_keys = ('block_signatures', 'feature_signatures', 'hint_signatures', 'read_signatures',
                   'hint_read_events', 'read_groups', 'cpu_counts', 'prepare_rng_signatures',
                   'sampling_rng_signatures', 'model_rng_signatures', 'initial_model_sha256',
                   'final_rng_sha256', 'feature_index')
    with tempfile.TemporaryDirectory(prefix='digit-dense-graph-') as folder:
        graph, arrays, bundle, features = fixture(folder, pin=True)
        try:
            for fanouts in FANOUT_CASES:
                baseline_result = baseline_state = None
                for variant in MODEL_VARIANTS:
                    result, state = exercise(graph, arrays, bundle, features, variant, fanouts)
                    if baseline_result is None:
                        baseline_result, baseline_state = result, state
                    else:
                        for key in parity_keys:
                            require(result[key] == baseline_result[key],
                                    'Group parity changed %s for %s/%s' % (key, fanouts, variant))
                        comparison = _compare_trees(baseline_state, state)
                        comparison['baseline'] = 'original'
                        comparison['variant'] = variant
                        comparison['fanouts'] = list(fanouts)
                        comparisons['%s:%s' % ('-'.join(map(str, fanouts)), variant)] = comparison
                    results.append(result)
                    print('Passed tiny group training: %s fanouts=%s updates=33 roots=129 tail=1' %
                          (variant, list(fanouts)), flush=True)
                    torch.cuda.empty_cache()
        finally:
            graph._graph.unpin_memory_()
    from .trace import kernel_evidence,compare_traces
    traces={mode:kernel_evidence(OUT/('tiny_'+mode+'_trace.json'),mode) for mode in VARIANTS}
    trace_comparison=compare_traces(traces['legacy'],traces['incremental'])
    require(tested_sources() == tested, 'Tested group source changed during GPU checks')
    require(sha(BINARY) == binary_sha256 and sha(SAMPLER_BINARY)==sampler_binary_sha256,
            'Bound binary changed during GPU checks')
    write(OUT / 'gpu_checks.json', dict(
        passed=True, model_variants=list(MODEL_VARIANTS), fanout_cases=[list(f) for f in FANOUT_CASES],
        results=results, comparisons=comparisons, model_updates=198, total_model_updates=198, tiny_trace_comparison=trace_comparison,
        exact_block_and_row_parity=True, exact_feature_parity=True, exact_model_rng_parity=True,
        model_and_adam_numerical_parity=True,
        model_and_adam_bit_exact=all(item['bit_exact'] for item in comparisons.values()),
        logits_and_final_gradients_compared=True, actual_loader_scheduler=True, io_mode='overlap',
        feature_reads_simulated=True, native_ticket_implementation_tested=False,
        raw_ssd_access=False, performance_evidence=False, fixture_dataloader_adapter=True,
        group_change="incremental lane weights",same_accepted_dense_graph=True,
        torch_version=torch.__version__, dgl_version=dgl.__version__,
        binary_sha256=binary_sha256, sampler_binary_sha256=sampler_binary_sha256,
        reference_sha256=reference_sha256, tested_sha256=tested,
    ))
    print('GPU original/legacy/incremental group training parity passed; simulated feature I/O, no SSD',
          flush=True)


if __name__ == '__main__':
    main()
