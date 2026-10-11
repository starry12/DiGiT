
import statistics
import math
from .config import MODES, digest
from .accounting import positive_time, validate_counts
from .order import order_policy

def aggregate(config, receipts):
    mode = MODES[config['execution']['name']]
    if len(receipts) != 2*mode.rounds:
        raise ValueError('All requested arm processes must complete')
    jobs = {}; workers = set()
    for receipt in receipts:
        r = receipt['report'];key = (r['repetition'], r['arm'])
        if key in jobs or key not in set(mode.schedule()):raise ValueError('Duplicate/unknown job')
        if receipt['returncode'] != 0 or not receipt['cleanup_verified'] or not r['passed'] or not r['complete']:
            raise ValueError('Unaccepted worker exit/cleanup')
        if r['initial']['evidence_scope']=='native' and not (receipt.get('admission_verified') and receipt.get('correctness_verified')):
            raise ValueError('Native admission/correctness acceptance missing')
        if r['config_sha256'] != digest(config) or r['mode'] != mode.name or r['dataset'] != config['dataset']:
            raise ValueError('Wrong configuration/report')
        if r.get('root_order_policy')!=order_policy(config,r['arm']):
            raise ValueError('Wrong arm root-order policy')
        if len(r['epochs']) != mode.epochs or [e['epoch'] for e in r['epochs']] != list(range(1, mode.epochs+1)):
            raise ValueError('Incomplete epochs')
        if r['full_epoch_training'] != (mode.name == 'full20'):
            raise ValueError('Wrong full-epoch scope')
        if not math.isclose(r['measured_seconds'],sum(e['seconds'] for e in r['epochs']),rel_tol=1e-12):
            raise ValueError('Epoch time total mismatch')
        if r['updates'] != sum(e['batches'] for e in r['epochs']) or r['final']['updates'] != r['updates']:
            raise ValueError('Update total mismatch')
        if r['final']['outstanding_io'] or r['final']['outstanding_prefetch'] or not r['final']['finite']:
            raise ValueError('Incomplete worker drain')
        for epoch in r['epochs']:
            positive_time(epoch['seconds'])
            validate_counts(epoch['counters'],config['data']['row_bytes'],r['arm'])
            validate_counts(epoch['complete_counters'],config['data']['row_bytes'],r['arm'])
            if mode.name=='full20':
                expected=(r['train_root_count']+config['batch_size']-1)//config['batch_size']
                if (epoch['roots']!=r['train_root_count'] or epoch['measured_roots']!=r['train_root_count']
                        or epoch['batches']!=expected or epoch['measured_batches']!=expected or epoch['warmup_seconds']!=0):
                    raise ValueError('Incomplete full training epoch/tail')
        worker = r['initial']['worker_id']
        if worker in workers:raise ValueError('Each arm/round requires a fresh worker')
        workers.add(worker);positive_time(r['measured_seconds']);jobs[key] = r
    reference=receipts[0]['report']['initial']
    for receipt in receipts:
        for key in ('source_sha256','initial_model_sha256','graph_sha256','split_sha256',
                    'canonical_sha256','cpu_feature_bytes','gpu_feature_bytes','evidence_scope'):
            if receipt['report']['initial'][key]!=reference[key]:
                raise ValueError('Cross-round configuration drift: '+key)
    pairs = []
    for (number,arm),r in jobs.items():
        reference=jobs[1,arm]
        for x,y in zip(r['epochs'],reference['epochs']):
            for key in ('roots_sha256','measured_roots_sha256'):
                if x[key]!=y[key]:raise ValueError('Cross-round root-order drift: '+arm)
    for number in range(1, mode.rounds+1):
        a, b = jobs[number, 'gids'], jobs[number, 'digit']
        for key in ('initial_model_sha256','source_sha256','graph_sha256','split_sha256',
                    'canonical_sha256','cpu_feature_bytes','gpu_feature_bytes','evidence_scope'):
            if a['initial'][key] != b['initial'][key]:raise ValueError('Unmatched '+key)
        for x,y in zip(a['epochs'], b['epochs']):
            for key in ('roots','measured_roots','batches','measured_batches'):
                if x[key] != y[key]:raise ValueError('Unmatched epoch '+key)
            if mode.measured_batches and (x['batches'] != 320 or x['measured_batches'] != 300):
                raise ValueError('Short window extent changed')
        pairs.append(dict(round=number,gids_seconds=a['measured_seconds'],digit_seconds=b['measured_seconds'],
                          speedup=a['measured_seconds']/b['measured_seconds'],
                          per_epoch_speedups=[x['seconds']/y['seconds'] for x,y in zip(a['epochs'],b['epochs'])]))
    scope = receipts[0]['report']['initial']['evidence_scope']
    return dict(passed=True, complete=True, dataset=config['dataset'], mode=mode.name,
                config_sha256=digest(config), paired_rounds=pairs,
                speedup=statistics.median(p['speedup'] for p in pairs),
                aggregation='median_of_five_paired_speedups' if mode.rounds==5 else 'single_pair_ratio_of_total_training_time',
                migration_acceptance_passed=mode.name=='short1' and scope=='native',
                evidence_scope=scope, stable_speedup_established=False,
                full_epoch_training=mode.name=='full20',
                matched_root_order=False,matched_short_window_examples=False,
                matched_full_training_set=True,
                root_order_by_arm={arm:order_policy(config,arm) for arm in ('gids','digit')},
                accuracy_evaluated=all(r['report']['accuracy_evaluated'] for r in receipts))

def run_series(config, launch_worker, save):

    receipts = [];save(dict(stage='starting', passed=False, complete=False, receipts=[]))
    try:
        for repetition, arm in MODES[config['execution']['name']].schedule():
            receipt = launch_worker(config, arm, repetition);receipts.append(receipt)
            save(dict(stage='worker_complete', passed=False, complete=False, receipts=list(receipts)))
            if receipt['returncode'] != 0 or not receipt['cleanup_verified'] or not receipt['report']['passed'] or not receipt['report']['complete']:
                raise RuntimeError('Stage failed: round %d %s' % (repetition, arm))
            if receipt['report']['initial']['evidence_scope']=='native' and not (receipt.get('admission_verified') and receipt.get('correctness_verified')):
                raise RuntimeError('Native admission/correctness acceptance missing')
        result = aggregate(config, receipts);save(dict(stage='complete', **result, receipts=receipts));return result
    except BaseException as error:
        save(dict(stage='failed', passed=False, complete=False, error=repr(error), receipts=receipts));raise
