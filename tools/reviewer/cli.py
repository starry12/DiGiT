#!/usr/bin/python3
"""Reviewer entry: fixed experiment routes, concise results and accepted references."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ADMIN = Path('/srv/digit-ae/admin')
HANDLERS = {
    'main': Path(__file__).with_name('pa_cli.py'),
    'ablation': ADMIN / 'ablation_v1/cli.py',
    'layout': ADMIN / 'layout_v1/cli.py',
    'IG': ADMIN / 'ig_performance_v1/cli.py',
    'UKS': ADMIN / 'uks_performance_v2/cli.py',
    'UKL': ADMIN / 'ukl_performance_v2/cli.py',
    'CL': ADMIN / 'cl_performance_v1/cli.py',
}
HANDLERS.update({dataset + '_' + model: ADMIN / 'multimodel_v2' / (dataset.lower() + '_' + model) / 'cli.py'
                 for dataset in ('IG', 'UKS', 'UKL', 'CL') for model in ('gcn', 'gat')})
INSPECT = ('status', 'results', 'logs', 'stop')


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, epilog='Activate with: source /srv/digit-ae/activate.sh')
    p.add_argument('command', choices=('check', 'smoke', 'run', 'ablation', 'layout', 'performance') + INSPECT)
    p.add_argument('dataset', choices=('PA', 'IG', 'UKS', 'UKL', 'CL'))
    p.add_argument('model', choices=('sage', 'gcn', 'gat'))
    p.add_argument('--action', choices=('check', 'smoke', 'run', 'ablation', 'layout', 'performance'))
    p.add_argument('--json', action='store_true')
    p.add_argument('--reference', action='store_true')
    a = p.parse_args(argv)
    if a.command not in INSPECT and a.action:
        p.error('Launch commands take no --action selector')
    selected = a.action if a.command in INSPECT else a.command
    if a.dataset == 'PA':
        if selected == 'performance':
            p.error('Use run, ablation or layout for PA')
        if selected in ('ablation', 'layout'):
            if a.model != 'sage':
                p.error('Ablation and layout support PA sage')
            a.route = selected
        else:
            a.route = 'main'
    else:
        if selected != 'performance':
            p.error('IG/UKS/UKL/CL use performance <dataset> <sage|gcn|gat>; inspection requires --action performance')
        a.route = a.dataset if a.model == 'sage' else a.dataset + '_' + a.model
    if a.reference and (a.command != 'results' or a.route not in ('layout', 'IG', 'UKS')):
        p.error('--reference is available for layout, IG and UKS results')
    if a.json and a.command not in ('status', 'results'):
        p.error('--json is available for status/results')
    return a


def ratio(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid accepted speedup')
    return '%.2f×' % value


def display(a, value):
    print('%s / %s | %s' % (a.dataset, a.model.upper(), value['state']))
    if value['state'] in ('PASS', 'AE_REFERENCE'):
        if a.route == 'main':
            for result in value.get('results', []):
                speedup = result.get('ratios', {}).get('training_speedup')
                if speedup is not None:
                    print('DiGiT vs GIDS: ' + ratio(speedup))
                for row in result.get('rows', []):
                    acc = row.get('test_accuracy')
                    if acc is not None:
                        if not isinstance(acc, (float, int)) or not math.isfinite(acc) or not 0 <= acc <= 1:
                            raise ValueError('Invalid test accuracy')
                        print('%s test accuracy: %.2f%%' % ('GIDS' if row['arm'] == 'gids' else 'DiGiT', acc * 100))
        elif a.route == 'ablation':
            print('DiGiT vs GIDS: ' + ratio(value['summary']['arms']['digit_full']['speedup_vs_gids']))
        elif a.route == 'layout':
            print('Best layout vs g2/r20: ' + ratio(max(row['speedup_vs_fresh_g2_r20'] for row in value['summary']['points'])))
        else:
            print('DiGiT vs GIDS: ' + ratio(value['summary']['speedup']))
    for name in ('error', 'output'):
        if value.get(name):
            print(name + ': ' + str(value[name]))


def uks_reference():
    relative = 'reference/uks_sage_reviewer_acceptance.json'
    path = ROOT / relative
    manifest = json.loads((ROOT / 'ARTIFACT_MANIFEST.json').read_text())
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][relative]:
        raise ValueError('AE reference identity changed')
    receipt = json.loads(path.read_text())
    if not receipt.get('passed') or not receipt.get('native_ae_replay_accepted') or receipt.get('formal_workers') != 10:
        raise ValueError('UKS AE reference not accepted')
    ratio(receipt['speedup'])
    return dict(state='AE_REFERENCE', summary=dict(speedup=receipt['speedup'], selection=receipt['selection']))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = parse(argv)
    if a.reference and a.route == 'UKS':
        value = uks_reference()
        print(json.dumps(value, indent=2)) if a.json else display(a, value)
        return 0
    if not HANDLERS[a.route].is_file():
        raise RuntimeError('This model service is not installed on the prepared server yet: ' + a.route)
    command = ['/usr/bin/python3', '-I', '-B', str(HANDLERS[a.route]), *argv]
    if a.command == 'results' and not a.json:
        result = subprocess.run(command + ['--json'], capture_output=True, text=True, timeout=120)
        if not result.stdout.strip():
            sys.stderr.write(result.stderr)
            return result.returncode or 1
        value = json.loads(result.stdout)
        display(a, value)
        if result.stderr:
            sys.stderr.write(result.stderr)
        return result.returncode
    os.execv(command[0], command)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
