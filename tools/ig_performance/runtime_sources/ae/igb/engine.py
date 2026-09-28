"""Standalone one-epoch GraphSAGE diagnostic with full validation."""
from ig_common import *
import random, ctypes, gc
import numpy as np, torch, dgl
from contract import TRAIN, VALID, TEST, BATCH, batch_slices, train_order, epoch_seed
from bounded_io import load_csc
from uva_sampler import UVANeighborSampler, native
from digit.artifacts import ArtifactBundle
from digit.sampler import DIGIT_STORAGE_ROW, DIGIT_STORAGE_IS_GROUP
from models import SAGE

def reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)

def step(model, opt, blocks, x, y):
    pred = model(blocks, x)
    loss = torch.nn.functional.cross_entropy(pred, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach()

def evaluate(model, sampler, graph, features, labels, roots, seed, provider, out, tag, small, mark):
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    was_training = model.training
    reset(seed)
    model.eval()
    guesses = hashlib.sha256()
    targets = hashlib.sha256()
    root_hash = hashlib.sha256()
    histogram = np.zeros(19, dtype=np.int64)
    per_class = np.zeros(19, dtype=np.int64)
    losses = []
    batch_records = []
    examples = correct = rows_seen = window_rows = 0
    before = None
    torch.cuda.synchronize()
    began = time.perf_counter()
    start_ns = time.monotonic_ns()
    with torch.no_grad():
        for i, rows in enumerate(roots):
            if provider and i % 100 == 0:
                before = provider.begin()
                window_rows = 0
            root = torch.from_numpy(rows.copy()).cuda()
            inputs, outputs, blocks = sampler.sample_blocks(graph, root)
            x = features(inputs, blocks, True)
            target = np.asarray(labels[rows]).astype('i8')
            y = torch.from_numpy(target).cuda()
            pred = model(blocks, x)
            loss = torch.nn.functional.cross_entropy(pred, y)
            prediction = pred.argmax(1).cpu().numpy()
            loss_value = float(loss)
            require(np.isfinite(loss_value), 'Nonfinite evaluation loss')
            hits = prediction == target
            correct += int(hits.sum())
            examples += len(rows)
            histogram += np.bincount(prediction, minlength=19)
            per_class += np.bincount(target[hits], minlength=19)
            guesses.update(memoryview(prediction).cast('B'))
            targets.update(memoryview(target).cast('B'))
            root_hash.update(memoryview(rows).cast('B'))
            losses.append((loss_value, len(rows)))
            rows_seen += len(inputs)
            window_rows += len(inputs)
            if small:
                batch_records.append(dict(batch=i, roots=digest(rows), predictions=digest(prediction), labels=digest(target), loss=loss_value, correct=int(hits.sum())))
            if provider and (i + 1) % 100 == 0:
                value = provider.finish(before, window_rows)
                append_sync(out / 'io_windows.jsonl', dict(phase=tag, batch_end=i + 1, **value))
                before = None
                progress(out, 'evaluating', phase=tag, batches=i + 1, examples=examples, total_batches=44362, evaluation_seconds=time.perf_counter() - began)
        if provider and before is not None:
            value = provider.finish(before, window_rows)
            append_sync(out / 'io_windows.jsonl', dict(phase=tag, batch_end=len(losses), **value))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    end_ns = time.monotonic_ns()
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    torch.cuda.set_rng_state_all(states[3])
    model.train(was_training)
    restored = torch.equal(torch.get_rng_state(), states[2]) and all((torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), states[3])))
    require(restored, 'Torch evaluation RNG restore failed')
    mark(tag + '_complete', examples=examples)
    return dict(examples=examples, batches=len(losses), correct=correct, accuracy=correct / examples, loss=sum((v * n for v, n in losses)) / examples, predictions_sha256=guesses.hexdigest(), labels_sha256=targets.hexdigest(), roots_sha256=root_hash.hexdigest(), prediction_histogram=histogram.tolist(), per_class_correct=per_class.tolist(), batch_records=batch_records, torch_rng_restored=restored, dgl_next_epoch_reseed=True, feature_rows=rows_seen, seconds=seconds, start_ns=start_ns, end_ns=end_ns)

def run(arm, mode, name, seed):
    require(mode == 'formal', 'Only the extracted one-epoch protocol is supported')
    require(arm in ('gids', 'full') and seed == 0, 'Two fresh one-epoch workers only')
    from sampler_config import configure
    sampler_configuration = configure(arm)
    require(native.get_optimized() == (arm == 'full') and (not native.profile_pending()), 'Wrong native variant or profiling state')
    execution = verify()
    out = Path(os.environ.get('DIGIT_RUN_OUTPUT', str(ROOT / 'results' / 'igb' / name)))
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'sampler_configuration.json', sampler_configuration)
    started = time.perf_counter()
    events = []

    def mark(stage, **kw):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        available = host()
        v = progress(out, stage, elapsed=time.perf_counter() - started, monotonic_ns=time.monotonic_ns(), gpu_free=free, host_available=available, allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), **kw)
        events.append(v)
        write(out / 'events.json', events)
        require(free >= 4 * GIB and available >= 64 * GIB, 'Runtime resource floor failed')
    try:
        require(host() >= 300 * GIB and (not observation()['pids']), 'Initial host/GPU admission failed')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.mem_get_info()[0] >= 42 * GIB, 'Initial GPU floor failed')
        torch.cuda.set_per_process_memory_fraction(24 * GIB / torch.cuda.get_device_properties(0).total_memory)
        mark('cuda_ready')
        provider = None
        from native_fetch import FormalFeatures
        provider = FormalFeatures(arm, out, mark)
        cm = read(L / 'csc/manifest.json')
        names = ['original_indptr.npy', 'original_indices.npy', 'original_eids.npy']
        graph, loaded = load_csc(*[L / 'csc' / x for x in names], N, host_cap=70 * GIB, expected_sha256=[cm['files'][x]['payload_sha256'] for x in names])
        ptrs = [x.data_ptr() for x in graph.adj_tensors('csc')]
        mark('normalized_csc_pinned')
        arrays = metadata = bundle = None
        if arm == 'full':
            manifest = read(L / 'full/manifest.json')
            arrays = {k: np.load(L / 'full' / v['path'], mmap_mode='r') for k, v in manifest['files'].items() if k != 'reordered_features'}
            bundle = ArtifactBundle(L / 'full', manifest, arrays)
            sampler = UVANeighborSampler([16, 5, 5], bundle, random_seed=seed, host_cap=100 * GIB)
            metadata = sampler._ensure_cuda_metadata(graph, torch.device('cuda:0'))
            require(metadata.accounting['original_csc_owned_bytes'] == 0 and metadata.accounting['host_logical_bytes'] == 86753666496, 'Metadata budget mismatch')
            write(out / 'metadata.json', metadata.accounting)
            mark('full_metadata_pinned')
        else:
            sampler = dgl.dataloading.NeighborSampler([16, 5, 5], replace=False)
        valid_sampler = dgl.dataloading.NeighborSampler([16, 5, 5], replace=False)
        raw = None
        labels = np.memmap(read(CFG / 'dataset.json')['dataset']['labels']['path'], dtype='<f4', mode='r', shape=(N,))
        if provider:
            provider.populate(None)
        reset(seed)
        model = SAGE(1024, 128, 19, 3, 0.2).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-08)
        initial = state_hash(model.state_dict())
        reset(seed)
        model.train()
        starting_updates = 0
        starting_checkpoint = None
        owners = (id(graph), id(sampler), id(metadata), id(model), id(opt), id(provider))
        mdptr = {k: v.data_ptr() for k, v in metadata.items()} if metadata else {}

        def retained():
            require(graph.is_pinned() and ptrs == [x.data_ptr() for x in graph.adj_tensors('csc')], 'Graph ownership changed')
            require(owners == (id(graph), id(sampler), id(metadata), id(model), id(opt), id(provider)), 'Persistent object changed')
            if metadata:
                require(mdptr == {k: v.data_ptr() for k, v in metadata.items()}, 'Metadata pointers changed')

        def features(inputs, blocks, standard=False):
            logical = inputs.cpu().numpy()
            if arm == 'full':
                physical = np.asarray(arrays['node_to_primary_row'][logical]).astype(np.int64) if standard else blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
                flags = np.zeros(len(logical), dtype=np.bool_) if standard else blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP].cpu().numpy()
            else:
                physical = logical
                flags = np.zeros(len(logical), dtype=np.bool_)
            x = torch.empty((len(logical), 1024), device='cuda')
            for at in range(0, len(logical), 16384):
                hi = min(len(logical), at + 16384)
                if provider:
                    x[at:hi].copy_(provider.fetch(physical[at:hi], flags[at:hi]))
                else:
                    x[at:hi].copy_(torch.from_numpy(raw.rows(logical[at:hi])))
            return x
        hooks = sum((len(getattr(m, k)) for m in model.modules() for k in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks')))
        require(hooks == 0 and getattr(opt.step, '__func__', None) is torch.optim.Adam.step, 'Unexpected hook/optimizer wrapper')
        mark('training_ready', initial_model=initial)
        epochs = []
        total_updates = starting_updates
        epoch_indices = range(read(CFG / 'protocol.json')['epochs'])
        for epoch in epoch_indices:
            retained()
            reset(epoch_seed(seed, epoch))
            model.train()
            torch.cuda.synchronize()
            began = time.perf_counter()
            order = train_order(seed, epoch)
            root_sha = digest(order)
            batches = (order[lo:hi] for lo, hi in batch_slices(TRAIN, boundary=False))
            order_seconds = time.perf_counter() - began
            iteration_start_ns = time.monotonic_ns()
            values = []
            pending = []
            windows = []
            before = None
            window_rows = total_rows = 0
            shape_totals = np.zeros((3, 3), dtype=np.int64)
            selected_root_hash = hashlib.sha256()
            count = 0
            for i, rows in enumerate(batches):
                if provider and i % 100 == 0:
                    before = provider.begin()
                    window_rows = 0
                root = torch.from_numpy(rows.copy()).cuda()
                inputs, outputs, blocks = sampler.sample_blocks(graph, root)
                x = features(inputs, blocks)
                y = torch.from_numpy(np.asarray(labels[rows]).astype('i8')).cuda()
                loss = step(model, opt, blocks, x, y)
                pending.append(loss)
                selected_root_hash.update(memoryview(rows).cast('B'))
                count += 1
                shape_totals += np.array([[b.num_src_nodes(), b.num_dst_nodes(), b.num_edges()] for b in blocks])
                total_rows += len(inputs)
                window_rows += len(inputs)
                if count % 100 == 0:
                    v = torch.stack(pending).cpu().numpy()
                    require(np.isfinite(v).all(), 'Nonfinite training losses')
                    values.extend(map(float, v))
                    pending.clear()
                    if provider:
                        io = provider.finish(before, window_rows)
                        append_sync(out / 'io_windows.jsonl', dict(phase='train_%02d' % epoch, batch_end=count, **io))
                        windows.append(io)
                        before = None
                    progress(out, 'training', epoch=epoch + 1, updates=count, total_updates=133085, order_seconds=order_seconds, iteration_seconds=(time.monotonic_ns() - iteration_start_ns) / 1000000000.0)
            if pending:
                v = torch.stack(pending).cpu().numpy()
                require(np.isfinite(v).all(), 'Nonfinite training losses')
                values.extend(map(float, v))
                pending.clear()
            if provider and before is not None:
                io = provider.finish(before, window_rows)
                append_sync(out / 'io_windows.jsonl', dict(phase='train_%02d' % epoch, batch_end=count, **io))
                windows.append(io)
            del x, loss, blocks, inputs, outputs
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - began
            iteration_end_ns = time.monotonic_ns()
            total_updates += count
            expected = (TRAIN + BATCH - 1) // BATCH
            require(count == expected and len(values) == expected, 'Training coverage mismatch')
            if arm == 'full':
                require(sampler._cuda_call_counter == total_updates, 'Full invocation counter mismatch')
            require(all((torch.isfinite(p).all().item() for p in model.parameters())), 'Nonfinite epoch model')
            losses = np.asarray(values, dtype='<f8')
            loss_path = out / ('losses_%02d.npy' % epoch)
            np.save(loss_path, losses)
            model_hash = state_hash(model.state_dict())
            opt_hash = state_hash(opt.state_dict())
            checkpoint = None
            del batches
            if not False:
                del order
            gc.collect()
            valid_roots = (np.arange(TRAIN + lo, TRAIN + hi, dtype=np.int64) for lo, hi in batch_slices(VALID, boundary=False))
            evaluation = evaluate(model, valid_sampler, graph, features, labels, valid_roots, 7001, provider, out, 'valid_%02d' % epoch, False, mark)
            expected_valid = VALID
            require(evaluation['examples'] == expected_valid, 'Validation coverage mismatch')
            retained()
            rec = dict(epoch=epoch, updates=count, total_optimizer_updates=total_updates, order_sha256=root_sha, selected_roots_sha256=selected_root_hash.hexdigest(), losses=binding(loss_path), loss_sha256=digest(losses), model=model_hash, optimizer=opt_hash, checkpoint=checkpoint, sampling_shape_totals=shape_totals.tolist(), feature_rows=total_rows, validation=evaluation, order_seconds=order_seconds, training_seconds=train_seconds, io_windows=len(windows), iteration_start_ns=iteration_start_ns, iteration_end_ns=iteration_end_ns)
            write(out / ('epoch_%02d.json' % epoch), rec)
            epochs.append(rec)
            mark('epoch_complete', epoch=epoch + 1, updates=count, training_seconds=train_seconds)
        test = None
        retained()
        if provider:
            provider.final_checks()
        final = out / 'final.pt'
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), final)
        mark('complete')
        require(verify() == execution, 'Execution changed')
        write(out / 'report.json', dict(passed=True, execution_sha256=execution, protocol_sha256=sha(CFG / 'protocol.json'), arm=arm, mode='formal', seed=seed, initial_model=initial, starting_checkpoint=starting_checkpoint, starting_updates=starting_updates, updates=total_updates - starting_updates, optimizer_updates=total_updates, epochs=epochs, test=test, final=binding(final), final_model=state_hash(model.state_dict()), final_optimizer=state_hash(opt.state_dict()), native=provider.receipt() if provider else None, metadata=binding(out / 'metadata.json') if metadata else None, hook_count=hooks, optimizer_step_unwrapped=True, persistent_objects=True, empty_cache_calls=0, per_batch_audits=False, reference_gradient_checks=False, source_feature_comparisons=False, route_warmup=False, training_seconds=sum((x['training_seconds'] for x in epochs)), validation_seconds=sum((x['validation']['seconds'] for x in epochs)), seconds=time.perf_counter() - started, formal_training=False, diagnostic_one_epoch=True, sampler_configuration=sampler_configuration, profile_mode='off', native_profile_armed=bool(native.profile_pending())))
    except BaseException as ex:
        import traceback
        write(out / 'failure.json', dict(error=repr(ex), traceback=traceback.format_exc(), time=time.time()))
        raise
