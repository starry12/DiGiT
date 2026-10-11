
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class Mode:
    name: str
    rounds: int
    epochs: int
    warmup_batches: int
    measured_batches: int

    def schedule(self):
        for repetition in range(1, self.rounds + 1):
            for arm in (('gids', 'digit') if repetition % 2 else ('digit', 'gids')):
                yield repetition, arm

MODES = {
    'short1': Mode('short1', 1, 1, 20, 300),
    'short5': Mode('short5', 5, 1, 20, 300),
    'full20': Mode('full20', 1, 20, 0, 0),
}

def load_config(dataset, mode='short1'):
    if dataset not in ('PA', 'IG', 'UKS', 'UKL', 'CL') or mode not in MODES:
        raise ValueError('Unknown dataset or mode')
    common = json.loads((HERE/'configs/common.json').read_text())
    data = json.loads((HERE/'configs/datasets.json').read_text())[dataset]
    value = dict(schema='digit-unified-training-v2', dataset=dataset, **common,
                 data=data, execution=asdict(MODES[mode]))
    short=dataset in ('IG','UKL','CL') and mode in ('short1','short5')
    value['cache_profile']=dict(scope='short_window' if short else 'full_epoch',batches=320 if short else None,seed=23,order_block_nodes=256 if short else 1024,fallback='existing_RevPR_set_then_logical_ID' if short else 'existing_full_epoch_policy')
    if short:
        value['profile_epochs']=0
        value['profile_order']=dict(policy='bfs_block_shuffle',block_nodes=256,seed=0)
    validate(value)
    return value

def validate(config):
    mode = MODES.get(config['execution']['name'])
    if mode is None or config['execution'] != asdict(mode):
        raise ValueError('Execution extent differs from named mode')
    d = config['data']
    if d['weight_decay'] != 0.0:
        raise ValueError('Unified v2 requires weight decay zero in both arms')
    if d['row_bytes'] != 4*d['feature_dim'] or d['classes'] <= 1:
        raise ValueError('Feature geometry/classes mismatch')
    if config['batch_size'] != 1024 or config['fanouts'] != [10, 5, 5]:
        raise ValueError('Unexpected training geometry')
    if config['window_buffer'] or config['accumulator'] or config['group_size'] != 2:
        raise ValueError('Unexpected cache/group protocol')
    if config['profile_seed'] == config['seed'] or config['gpu_cache_bytes'] != 4*2**30:
        raise ValueError('Profile independence/cache capacity mismatch')
    if config['order']['policy'] != 'gids_random_digit_bfs_chunk_shuffle' or config['order']['block_nodes'] != 256 or config['order']['seed'] != config['seed'] or config['order']['tail'] != 'retain':
        raise ValueError('Unexpected root order')
    short=config['dataset'] in ('IG','UKL','CL') and config['execution']['name'] in ('short1','short5')
    if config['profile_epochs'] != (0 if short else 1) or config['cpu_cache_fraction'] != 0.1:
        raise ValueError('Unexpected cache selection scope')
    if config['profile_order'] != dict(policy='bfs_block_shuffle',block_nodes=256 if short else 1024,seed=0):
        raise ValueError('Unexpected cache-selection order')
    expected=dict(scope='short_window' if short else 'full_epoch',batches=320 if short else None,seed=23,order_block_nodes=256 if short else 1024,fallback='existing_RevPR_set_then_logical_ID' if short else 'existing_full_epoch_policy')
    if config.get('cache_profile')!=expected:raise ValueError('Unexpected cache profiling scope')

def arm_config(config, arm):
    if arm not in ('gids', 'digit'):
        raise ValueError('Unknown arm')
    row = config['data']['row_bytes']
    return dict(arm=arm, gpu_policy='legacy' if arm == 'gids' else 'fifo',
                cpu_policy='per_node_revpr' if arm == 'gids' else 'full_epoch_frequency',
                sampling='uniform' if arm == 'gids' else 'strict_group_first',
                request_bytes=[row] if arm == 'gids' else [row, 2*row],
                async_prefetch=arm == 'digit', model_optimization=arm == 'digit')

def plan(dataset, mode='short1'):
    c = load_config(dataset, mode)
    return dict(config=c, config_sha256=digest(c),
                jobs=[dict(repetition=r, **arm_config(c, a)) for r, a in MODES[mode].schedule()],
                migration_acceptance_mode='short1',
                native_adapter_status='pending_integration', native_execution_enabled=False,
                aggregation=('median_of_five_paired_speedups' if mode == 'short5' else
                             'single_paired_speedup' if mode == 'short1' else 'ratio_of_20_epoch_totals'),
                full_epoch_warmup_policy='no_discarded_or_extra_training_batches',
                checkpoints='epoch-boundary export; exact dynamic-cache resume not yet supported')
