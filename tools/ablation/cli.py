"""Fixed PA/SAGE four-arm AE service: no caller-supplied privileged paths."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
CONTROL = Path('/srv/digit-ae/admin/ablation_v1')
OUTPUTS = Path('/srv/digit-ae/ablation-results')
UNIT = 'digit-ae-pa-sage-ablation.service'

def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['ablation', 'status', 'results', 'logs', 'stop'])
    p.add_argument('dataset', choices=['PA'])
    p.add_argument('model', choices=['sage'])
    p.add_argument('--action', dest='selector', choices=['ablation'])
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    if a.action != 'ablation' and a.selector != 'ablation': p.error('Specify --action ablation')
    if a.action == 'ablation' and a.selector: p.error('Start takes no --action selector')
    if a.json and a.action not in ('status', 'results'): p.error('--json is for status/results')
    return a

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked_output(value):
    p = Path(value)
    if not p.is_absolute() or p.parent != OUTPUTS or p.is_symlink() or p.resolve().parent != OUTPUTS.resolve():
        raise ValueError('Invalid result directory')
    return p

def view():
    latest = CONTROL / 'state/latest.json'
    if not latest.exists(): return dict(state='NOT_STARTED')
    request = read(latest); output = checked_output(request['output'])
    state = read(output / 'status.json')
    svc = subprocess.run(['/usr/bin/systemctl', 'show', UNIT, '--property=ActiveState,SubState,Result,ExecMainStatus'], text=True, capture_output=True, check=True)
    service = dict(line.split('=', 1) for line in svc.stdout.splitlines() if '=' in line)
    active = service.get('ActiveState') in ('active', 'activating', 'deactivating')
    kind = 'RUNNING' if active else 'INCOMPLETE'
    if state.get('stage') == 'failed': kind = 'FAILED'
    elif state.get('complete') and state.get('passed'):
        kind = 'FINALIZING' if active else ('PASS' if service.get('Result') == 'success' and service.get('ExecMainStatus') == '0' else 'FAILED')
    result = dict(state=kind, output=str(output), stage=state.get('stage'), service=service,
        workers=state.get('workers', []), error=state.get('error'), request=request)
    if kind == 'PASS':
        summary = read(output / 'full_summary.json')
        if not summary.get('passed') or sha(output / 'full_summary.json') != state.get('summary_sha256'):
            raise ValueError('Summary missing or changed')
        for arm, digest in summary['report_sha256'].items():
            if sha(output / ('full_' + arm + '_accepted.json')) != digest: raise ValueError('Accepted report changed')
        result['summary'] = summary
    for w in reversed(state.get('workers', [])):
        progress = output / w['mode'] / w['arm'] / 'progress.json'
        if w['status'] == 'running' and progress.exists(): result['progress'] = read(progress); break
    return result

def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    if a.action in ('ablation', 'stop'):
        subprocess.run(['/usr/bin/sudo', '-n', '/usr/bin/systemctl', '--no-block', 'start' if a.action == 'ablation' else 'stop', UNIT], check=True)
        print('Start requested; acceptance pending.' if a.action == 'ablation' else 'Stop requested; wait for resource release.')
        print('digit-ae status PA sage --action ablation')
        return 0
    v = view()
    if a.action == 'logs':
        if 'output' in v:
            output = checked_output(v['output'])
            for name in ['status.json', 'launcher.log'] + [w['mode'] + '_' + w['arm'] + '.log' for w in v['workers'][-1:]]:
                f = output / name
                if f.is_file(): subprocess.run(['/usr/bin/tail', '-n', '60', '--', str(f)], check=True)
    elif a.json: print(json.dumps(v, indent=2))
    else:
        print('PA / SAGE ablation | ' + v['state'])
        for k in ('stage', 'output', 'error', 'progress'):
            if v.get(k): print(k + ': ' + str(v[k]))
        if a.action == 'results' and 'summary' in v:
            for arm, row in v['summary']['arms'].items():
                print('%-12s %.2f s  %.4fx' % (arm, row['training_seconds'], row['speedup_vs_gids']))
            print(v['summary']['scope'])
    return (0 if v['state'] == 'PASS' else 1 if v['state'] == 'FAILED' else 3) if a.action == 'results' else 0

if __name__ == '__main__':
    try: sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr); sys.exit(1)
