import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ADMIN, DATASETS, MODES, OUT, read, unit, view

def route(argv):
    old = '--legacy' in argv or '--reference' in argv
    if old:
        if '--mode' in argv or any(x.startswith('--mode=') for x in argv):
            raise ValueError('Legacy/reference results do not take --mode')
        if '--legacy' in argv and (not argv or argv[0] not in ('status', 'results', 'logs')):
            raise ValueError('--legacy is only for inspecting historical results')
        return None
    if len(argv) < 3:
        return None
    action, dataset, model = argv[:3]
    selected = None
    for i, arg in enumerate(argv):
        if arg.startswith('--action='):
            selected = arg.split('=', 1)[1]
        elif arg == '--action' and i + 1 < len(argv):
            selected = argv[i + 1]
    if model == 'sage' and dataset in DATASETS and (action == 'performance' or
            (action in ('status', 'results', 'logs', 'stop') and selected == 'performance')):
        return dataset
    return None

def display(value, dataset, mode):
    print(dataset + ' / SAGE / ' + mode + ' | ' + value['state'])
    for key in ('reason', 'error', 'output'):
        if value.get(key):
            print(key + ': ' + str(value[key]))
    if 'progress' in value:
        print('progress: ' + __import__('json').dumps(value['progress'], ensure_ascii=False))
    if value['state'] != 'PASS':
        return
    s = value['summary']
    print('DiGiT vs GIDS: %.3fx (%s)' % (s['speedup'], s['aggregation']))
    print('Round  GIDS (s)  DiGiT (s)  Speedup')
    for p in s['paired_rounds']:
        print('%5d %9.3f %10.3f %8.3fx' % (p['round'], p['gids_seconds'], p['digit_seconds'], p['speedup']))
    for arm in ('gids', 'digit'):
        a = s['arms'][arm]; c = a['measured_counters']; m = a['weighted_metrics']
        print('%s: CPU %.2f%%; GPU %.2f%%; grouped primary commands %s; total SSD commands %s; SSD bytes %s' %
              (arm.upper(), 100*m['cpu_service_fraction'], 100*m['gpu_service_fraction'],
               c['primary_pair_commands'], c['completed_commands'], c['completed_bytes']))
        if s['accuracy_evaluated']:
            print(arm.upper() + ' final test: ' + str(a.get('test')))
    print('Root order: ' + s.get('root_order_comparison', 'see summary.json'))
    if s.get('short_profile_functional_only'):
        print('320-batch profiling: functional check; not full-epoch cache-effectiveness evidence.')
    print('Full counters and epoch rows: summary.json / summary.csv in output directory.')

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv in (['--help'], ['-h']):
        print('Unified GraphSAGE: digit-ae performance {PA,IG,UKS,UKL,CL} sage [--mode short1|short5|full20]')
        print('Default: short5 (five pairs, median). Inspect: status/results/logs/stop DATASET sage --action performance [--mode ...]')
        print('Readiness: digit-ae unified; old results: results DATASET sage --action performance --legacy')
        print('The original PA run entry is retained. GCN/GAT and layout launch gates remain in effect.')
        return 0
    if argv == ['unified']:
        registry = read(ADMIN / 'registry.json')
        for d in DATASETS:
            row = registry['datasets'][d]
            print(d + ': ' + ('OPEN' if row['enabled'] else 'PENDING_ACCEPTANCE') + ' | ' + row['reason'])
        return 0
    dataset = route(argv)
    if dataset is None:
        if '--mode' in argv or any(x.startswith('--mode=') for x in argv):
            raise ValueError('--mode only applies to unified GraphSAGE performance')
        args = [x for x in argv if x != '--legacy']
        os.execv('/usr/bin/python3', ['/usr/bin/python3', '-I', '-B', str(ADMIN/'legacy_gate.py'), *args])
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('performance', 'status', 'results', 'logs', 'stop'))
    p.add_argument('dataset', choices=DATASETS)
    p.add_argument('model', choices=('sage',))
    p.add_argument('--action', choices=('performance',))
    p.add_argument('--mode', choices=MODES, default='short5')
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    if (a.command == 'performance') == bool(a.action):
        p.error('Launch uses performance; inspection/stop requires --action performance')
    if a.json and a.command not in ('status', 'results'):
        p.error('--json requires status/results')
    row = read(ADMIN/'registry.json')['datasets'][dataset]
    if a.command == 'performance' and not row['enabled']:
        print('PENDING_ACCEPTANCE: ' + row['reason'], file=sys.stderr)
        return 69
    if a.command in ('performance', 'stop'):
        if a.command == 'performance':
            print('Validation status: ' + row.get('reason', 'See digit-ae unified'), flush=True)
        subprocess.run(['/usr/bin/sudo', '-n', '/usr/bin/systemctl', '--no-block',
            'start' if a.command == 'performance' else 'stop', unit(dataset, a.mode)], check=True)
        print('Request sent. Check digit-ae status %s sage --action performance --mode %s' % (dataset, a.mode))
        return 0
    value = view(dataset, a.mode)
    if value['state'] == 'NOT_STARTED' and not row['enabled']:
        value.update(state='PENDING_ACCEPTANCE', reason=row['reason'])
    if a.command == 'logs' and value.get('output'):
        subprocess.run(['/usr/bin/tail', '-n', '60', '--', str(Path(value['output'])/'controller.log')], check=True)
    elif a.json:
        print(__import__('json').dumps(value, indent=2))
    else:
        display(value, dataset, a.mode)
    return 1 if value['state'] in ('FAILED', 'STALE') else 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
