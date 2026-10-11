"""Add disjoint logical service routes and actual primary I/O size counts."""
import math

FIELDS = ('logical_rows', 'cpu_rows', 'gpu_rows', 'ssd_rows',
          'primary_single_commands', 'primary_pair_commands', 'primary_bytes',
          'replay_commands', 'replay_bytes', 'completed_commands', 'completed_bytes',
          'sampled_group_edges', 'sampled_outer_edges')


def validate_counts(c, row_bytes, arm):
    if set(c) != set(FIELDS) or any(type(c[k]) is not int or c[k] < 0 for k in FIELDS):
        raise ValueError('Incomplete or invalid window counters')
    if c['logical_rows'] != c['cpu_rows']+c['gpu_rows']+c['ssd_rows']:
        raise ValueError('CPU/GPU/SSD routes do not reconcile')
    single, pair = c['primary_single_commands'], c['primary_pair_commands']
    if c['primary_bytes'] != row_bytes*(single+2*pair):
        raise ValueError('Primary command sizes/bytes mismatch')
    if c['completed_commands'] != single+pair+c['replay_commands'] or c['completed_bytes'] != c['primary_bytes']+c['replay_bytes']:
        raise ValueError('Completed/replay I/O mismatch')
    if arm == 'gids' and pair:
        raise ValueError('GIDS issued grouped reads')
    if c['sampled_group_edges'] > c['sampled_outer_edges']:
        raise ValueError('Invalid grouped sampling counts')


def add_counts(a, b):
    return {k: a.get(k, 0)+b[k] for k in FIELDS}


def metrics(c):
    n = c['logical_rows']; commands = c['primary_single_commands']+c['primary_pair_commands']
    return dict(cpu_service_fraction=c['cpu_rows']/n if n else 0,
                gpu_service_fraction=c['gpu_rows']/n if n else 0,
                ssd_service_fraction=c['ssd_rows']/n if n else 0,
                pair_command_fraction=c['primary_pair_commands']/commands if commands else 0,
                sampled_group_edge_fraction=c['sampled_group_edges']/c['sampled_outer_edges'] if c['sampled_outer_edges'] else 0)


def positive_time(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid measured time')
