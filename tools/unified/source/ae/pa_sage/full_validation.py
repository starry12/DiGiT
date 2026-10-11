"""Acceptance for each frozen PA seed/repeat pair; never infer completion from logs."""
import hashlib
import math


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def io_check(value, arm):
    device = value['device']
    require(device['enabled'] and device['reconciled'], 'Unreconciled device I/O')
    require(device['submitted_commands'] == device['completed_commands'], 'Unfinished I/O')
    require(value['gpu']['capacity_pages'] == 1048576 and value['gpu']['reconciled'], 'Wrong cache capacity/counters')
    require(value['reconciliation']['reconciled'], 'Missing I/O reconciliation')
    require(value['gpu']['requests'] == value['feature']['gpu_ssd'], 'Feature/cache count mismatch')
    if arm == 'digit_full':
        require(value['mixed']['enabled'] and value['mixed']['reconciled'], 'Missing mixed I/O')
        require(value['reconciliation']['device_counters_checked'], 'Missing native counter check')


def evaluation_check(value, split, contract, arm):
    count = contract[split+'_examples']
    require(value['examples'] == count and value['batches'] == math.ceil(count/contract['batch_size']), 'Incomplete '+split)
    require(value['trace_manifest_sha256'] == contract[split+'_trace_sha256'], 'Wrong '+split+' trace')
    require(value['training_rng_restored'], 'Evaluation changed training RNG')
    require(math.isfinite(value['loss']) and math.isfinite(value['seconds']) and value['seconds'] > 0, 'Invalid evaluation loss/time')
    require(0 <= value['accuracy'] <= 1, 'Invalid accuracy')
    histogram, correct = value['prediction_histogram'], value['per_class_correct']
    require(len(histogram) == len(correct) == contract['classes'], 'Wrong class count')
    require(all(type(x) is int and x >= 0 for x in histogram+correct), 'Invalid class counters')
    require(sum(histogram) == count and sum(correct) <= count, 'Incomplete evaluation histogram')
    require(close(value['accuracy'], sum(correct)/count), 'Accuracy/correct-count mismatch')
    io_check(value, arm)


def validate_report(report, arm, binding, contract, native_receipt_sha256):
    require(report['passed'] is True and report['smoke'] is False and report['source_only'] is False, 'Not a completed native full run')
    require(report['arm'] == arm and report['seed'] == contract['seed'], 'Wrong arm/seed')
    require(report['raw_writes'] is False and report['bfs_enabled'] is False and report['paired_random_order'], 'Wrong I/O/order contract')
    for key, expected in binding.items():
        require(report[key] == expected, 'Execution binding changed: '+key)
    native = report['native_validation_at_start']
    require(native['passed'] and native['receipt_sha256'] == native_receipt_sha256, 'Wrong native smoke prerequisite')
    require(report['admission']['passed'] and report['admission']['profile_sha256'] == binding['admission_profile_sha256'], 'Admission failed/changed')
    require(report['cache_bytes'] == 4*2**30 and report['cpu_cache_rows'] == contract['cpu_cache_rows'], 'Cache budget changed')
    require(report['all_epochs_metadata_and_cache_reused'], 'Persistent training state lost')
    epochs = report['epochs']
    batches = math.ceil(contract['train_examples']/contract['batch_size'])
    require(contract['epochs'] == 20 and len(epochs) == 20 and report['updates'] == 20*batches, 'Incomplete 20-epoch training')
    require(report['repeat']==contract['repeat'], 'Wrong repeat')
    lifecycle=report['evaluation_lifecycle']
    require(lifecycle['mode']=='formal_single_pass' and lifecycle['validation_calls']==20 and lifecycle['test_calls']==1 and lifecycle['diagnostic_replays']==0 and not lifecycle['cpu_diagnostic_monitor'] and lifecycle['training_rng_unchanged'], 'Unexpected evaluation replay/count')
    require(lifecycle['trace_preparation_seconds']>0, 'Missing trace preparation time')
    training_seconds = validation_seconds = groups = targets = shortfall = 0
    for number, epoch in enumerate(epochs, 1):
        require(epoch['epoch'] == number and epoch['updates'] == batches, 'Wrong epoch/update count')
        require(epoch['root_sha256'] == contract['root_sha256'][number-1], 'Wrong training root order')
        require(epoch['metadata_reused'] and epoch['cache_object_reused'], 'Epoch state was re-created')
        require(len(epoch['losses']) == batches and all(math.isfinite(x) for x in epoch['losses']), 'Missing/nonfinite training losses')
        shapes = epoch['shapes']
        require(len(shapes) == batches, 'Missing training batches')
        require(all(s['output_nodes'] == contract['batch_size'] for s in shapes[:-1]), 'Truncated training batch')
        require(shapes[-1]['output_nodes'] == contract['train_examples']-(batches-1)*contract['batch_size'], 'Final partial batch lost')
        require(sum(s['output_nodes'] for s in shapes) == contract['train_examples'], 'Training split coverage mismatch')
        require(math.isfinite(epoch['train_seconds']) and 0 <= epoch['order_seconds'] <= epoch['train_seconds'] and epoch['train_seconds'] > 0, 'Invalid epoch timing')
        cursor = 1
        for window in epoch['windows']:
            require(window['start_update'] == cursor and cursor <= window['end_update'] <= batches, 'Gapped/duplicate training window')
            cursor = window['end_update']+1
            require(0 <= window['actual_edges'] <= window['target_edges'], 'Invalid sampling coverage')
            require(window['target_edges']-window['actual_edges'] == window['shortfall_edges'], 'Incorrect sampling shortfall')
            groups += window['group_edges']; targets += window['target_edges']; shortfall += window['shortfall_edges']
            io_check(window, arm)
        require(cursor == batches+1, 'Training windows incomplete')
        io_check(epoch['training'], arm)
        evaluation_check(epoch['validation'], 'valid', contract, arm)
        training_seconds += epoch['train_seconds']
        validation_seconds += epoch['validation']['seconds']
    require(groups > 0 if arm == 'digit_full' else groups == 0, 'Wrong grouped sampling path')
    require(close(report['training_seconds'], training_seconds) and close(report['validation_seconds'], validation_seconds), 'Timing totals mismatch')
    require(close(report['mean_training_epoch_seconds'], training_seconds/20), 'Mean excludes/miscounts epochs')
    require(report['test'] is not None, 'Final test missing')
    evaluation_check(report['test'], 'test', contract, arm)
    require(report['final_parameters_sha256'] == epochs[-1]['model_sha256'], 'Final test checkpoint is not epoch 20')
    for cache in (report['initial_cache'], report['final_cache']):
        require(cache['gpu']['capacity_pages'] == 1048576 and cache['device']['outstanding'] == 0, 'Cache size/outstanding I/O mismatch')
    totals=report['timing_totals'];online=training_seconds+validation_seconds+report['test']['seconds']
    require(close(totals['training_validation_test_seconds'],online) and close(totals['with_trace_preparation_seconds'],online+lifecycle['trace_preparation_seconds']) and totals['diagnostic_control_seconds']==0, 'Incorrect evaluation/preparation timing')
    return dict(seed=report['seed'],repeat=report['repeat'],trace_preparation_seconds=lifecycle['trace_preparation_seconds'],online_seconds=online,with_trace_preparation_seconds=totals['with_trace_preparation_seconds'],updates=report['updates'], epochs=20, train_examples_per_epoch=contract['train_examples'],
        validation_examples_per_epoch=contract['valid_examples'], test_examples=contract['test_examples'],
        mean_training_epoch_seconds=training_seconds/20, training_seconds=training_seconds,
        order_excluded_mean_training_seconds=sum(e['train_seconds']-e['order_seconds'] for e in epochs)/20,
        validation_seconds=validation_seconds, test_seconds=report['test']['seconds'],
        test_accuracy=report['test']['accuracy'], test_correct=sum(report['test']['per_class_correct']),
        group_edges=groups, target_edges=targets, shortfall_edges=shortfall,
        group_edge_fraction=groups/targets if targets else 0,
        peak_torch_allocated_bytes=report['peak_gpu_allocated_bytes'], peak_host_rss_bytes=report['peak_host_rss_kib']*1024)


def validate_pair(reports):
    require(set(reports) == {'gids','digit_full'}, 'Missing paired arm')
    a,b = reports['gids'],reports['digit_full']
    require(a['initial_parameters_sha256'] == b['initial_parameters_sha256'], 'Unpaired model initialization')
    require(a['initial_dgl_rng'] == b['initial_dgl_rng'], 'Unpaired initial sampler RNG')
    require([e['root_sha256'] for e in a['epochs']] == [e['root_sha256'] for e in b['epochs']], 'Unpaired root orders')
    return dict(digit_speedup=a['mean_training_epoch_seconds']/b['mean_training_epoch_seconds'],
                digit_training_time_change_percent=(b['training_seconds']/a['training_seconds']-1)*100,
                digit_test_accuracy_difference_pp=(b['test']['accuracy']-a['test']['accuracy'])*100,
                scope='One paired seed/repeat; aggregate repeats within each seed separately')


def checkpoint_hash(path):
    import torch
    weights = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(weights, dict) and weights, 'Empty or invalid saved checkpoint')
    digest = hashlib.sha256()
    for name,tensor in weights.items():
        require(torch.is_tensor(tensor) and torch.isfinite(tensor).all().item(), 'Nonfinite/invalid checkpoint')
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
