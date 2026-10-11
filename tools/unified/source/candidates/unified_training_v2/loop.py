

import hashlib
import time
from .config import digest
from .order import windows, roots_digest, order_policy
from .accounting import validate_counts, add_counts, positive_time, metrics, FIELDS

class Adapter:

    def initialize(self, config, arm):

        raise NotImplementedError

    def run_window(self, batches, epoch, phase):

        raise NotImplementedError

    def epoch_end(self, epoch):

        return dict(evaluated=False, checkpoint=None)

    def finish(self):

        raise NotImplementedError

    def close(self):

        raise NotImplementedError

def run_arm(config, arm, repetition, canonical, adapter, emit=lambda event: None):
    if arm not in ('gids', 'digit') or not 1 <= repetition <= config['execution']['rounds']:
        raise ValueError('Invalid job')
    started = time.perf_counter(); epochs = []; updates = 0
    try:
        initial = adapter.initialize(config, arm)
        required = ('initial_model_sha256', 'source_sha256', 'graph_sha256', 'split_sha256',
                    'canonical_sha256', 'cpu_feature_bytes', 'gpu_feature_bytes', 'worker_id', 'evidence_scope')
        if any(k not in initial for k in required) or not initial.get('roots_verified_exact') or initial.get('gpu_resident_rows_start') != 0:
            raise ValueError('Missing input/cold-cache acceptance')
        if initial['evidence_scope'] not in ('native', 'cpu_fixture') or initial['gpu_feature_bytes'] != config['gpu_cache_bytes']:
            raise ValueError('Invalid evidence scope/cache budget')

        canonical_hash = roots_digest(canonical[i:i+1048576] for i in range(0, len(canonical), 1048576))
        if canonical_hash != initial['canonical_sha256']:
            raise ValueError('Prepared canonical roots changed')
        setup_seconds = time.perf_counter()-started
        for epoch in range(config['execution']['epochs']):
            wall = time.perf_counter(); measured = 0.; warmup = 0.; root_count = 0; batch_count = 0
            all_counts = {k: 0 for k in FIELDS}; timed_counts = dict(all_counts)
            epoch_hash = hashlib.sha256(); timed_hash = hashlib.sha256(); timed_roots = timed_batches = 0
            for index, (phase, batches) in enumerate(windows(canonical, config, epoch, arm)):
                expected = roots_digest(batches); roots = sum(len(b) for b in batches)
                r = adapter.run_window(batches, epoch, phase)
                if (r['updates'] != len(batches) or r['roots'] != roots or r['roots_sha256'] != expected
                        or not r['finite'] or r['outstanding_io'] or r['outstanding_prefetch']):
                    raise ValueError('Incomplete/mismatched window or undrained work')
                positive_time(r['seconds']);validate_counts(r['counters'], config['data']['row_bytes'], arm)
                all_counts = add_counts(all_counts, r['counters'])
                for batch in batches:
                    epoch_hash.update(batch.astype('<i8', copy=False).tobytes())
                    if phase == 'training':timed_hash.update(batch.astype('<i8', copy=False).tobytes())
                batch_count += len(batches);root_count += roots;updates += len(batches)
                if phase == 'training':
                    measured += r['seconds'];timed_counts = add_counts(timed_counts, r['counters'])
                    timed_roots += roots;timed_batches += len(batches)
                else:warmup += r['seconds']
                emit(dict(stage='window', epoch=epoch+1, window=index, phase=phase, **r))
            mode = config['execution']
            expected_roots = (mode['warmup_batches']+mode['measured_batches'])*config['batch_size'] if mode['measured_batches'] else len(canonical)
            if root_count != expected_roots:
                raise ValueError('Incomplete epoch/root coverage')

            training_wall = time.perf_counter()-wall
            auxiliary = adapter.epoch_end(epoch)
            if config['data']['labels'] == 'synthetic' and auxiliary.get('evaluated'):
                raise ValueError('Synthetic labels cannot provide accuracy evidence')
            e = dict(epoch=epoch+1, batches=batch_count, roots=root_count, measured_batches=timed_batches,
                     measured_roots=timed_roots, roots_sha256=epoch_hash.hexdigest(),
                     measured_roots_sha256=timed_hash.hexdigest(), seconds=measured, warmup_seconds=warmup,
                     training_wall_seconds=training_wall, counters=timed_counts, complete_counters=all_counts,
                     metrics=metrics(timed_counts), auxiliary=auxiliary)
            epochs.append(e);emit(dict(stage='epoch_complete', **e))
        final = adapter.finish()
        if (final['updates'] != updates or final['outstanding_io'] or final['outstanding_prefetch']
                or not final['finite'] or final['final_model_sha256'] == initial['initial_model_sha256']):
            raise ValueError('Incomplete training/model did not update')
        result = dict(passed=True, complete=True, dataset=config['dataset'], mode=config['execution']['name'],
                      arm=arm, repetition=repetition, config_sha256=digest(config), initial=initial, final=final,
                      root_order_policy=order_policy(config,arm),
                      epochs=epochs, updates=updates, train_root_count=len(canonical), setup_seconds=setup_seconds,
                      measured_seconds=sum(e['seconds'] for e in epochs),
                      accuracy_evaluated=any(e['auxiliary'].get('evaluated', False) for e in epochs),
                      full_epoch_training=not bool(config['execution']['measured_batches']))
    finally:
        adapter.close()
    return result
