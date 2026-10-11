import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

def source_binding():
    files = {}
    visited = set()
    def visit(path):
        path = Path(path)
        if path in visited:
            return
        visited.add(path)
        files[str(path)] = sha(path)
        if path.name == 'manifest.json':
            for rel, expected in read(path)['files'].items():
                target = ROOT/rel
                if not target.is_file():
                    target = path.parent/rel
                require(inside(target, ROOT), 'Invalid manifest member')
                require(sha(target) == expected, 'Source manifest drift: ' + str(target))
                visit(target)
    visit(ENTRY)
    for path in (ROOT/'digit_paths.py', ROOT/'candidates/pa_sage_bfs256_full20_v1/manifest.json',
                 ROOT/'candidates/pa_sage_bfs256_short_v1/manifest.json'):
        require(path.is_file(), 'Apply and verify PA BFS256 dispatch before opening unified entries')
        visit(path)
    return dict(entry_sha256=sha(ENTRY), files=files)

def acceptance(dataset, folder):
    folder = Path(folder).resolve()
    require(folder.parent == ROOT/'results/unified_experiments_20261011_v2'/dataset/'runs', 'Use an original unified short1 run directory')
    state, plan, summary = (read(folder/name) for name in ('status.json', 'plan.json', 'summary.json'))
    require(plan['dataset'] == dataset and plan['mode'] == 'short1', 'One accepted short1 pair required')
    require(state['stage'] == 'complete' and state['passed'] and state['complete'], 'Run is not complete and accepted')
    validate_result(summary, dataset, 'short1')
    require(state['source_sha256'] == plan['source_sha256'] == sha(ENTRY),
            'Historical pass uses a different entry version; validate current dispatch before opening')
    require(plan['root_order_comparison'] == 'GIDS_random_DiGiT_BFS256', 'BFS256 dispatch required')
    expected_unit = 'digit-unified-v2-' + dataset.lower() + '-' + folder.name
    require(plan['unit'] == expected_unit, 'Unexpected acceptance service')
    text = subprocess.check_output(['/usr/bin/systemctl', 'show', expected_unit + '.service',
        '--property=ActiveState,Result,ExecMainStatus'], text=True, timeout=15)
    service_state = dict(x.split('=', 1) for x in text.splitlines() if '=' in x)
    require(not active(service_state) and service_state.get('Result') == 'success' and service_state.get('ExecMainStatus') == '0',
            'Acceptance service must have exited successfully')
    return dict(output=str(folder), summary_sha256=sha(folder/'summary.json'),
                status_sha256=sha(folder/'status.json'), plan_sha256=sha(folder/'plan.json'), service=service_state)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--run', type=Path)
    p.add_argument('--disable', action='store_true')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.disable == bool(a.run):
        p.error('Specify either --run for acceptance or --disable')
    if a.disable:
        evidence, binding = None, None
    else:
        evidence = acceptance(a.dataset, a.run)
        binding = source_binding()
        verify_binding(binding)
    if not a.execute:
        print(__import__('json').dumps(dict(dataset=a.dataset, enabled=not a.disable, acceptance=evidence,
            source_sha256=binding['entry_sha256'] if binding else None, starts_experiments=False), indent=2))
        return
    require(os.geteuid() == 0, 'Use sudo for the root-owned launch registry')
    from install import checks
    checks()
    for mode in MODES:
        require(not active(service(a.dataset, mode)), 'Dataset service is active; do not change its binding')
    registry = read(ADMIN/'registry.json')
    marker = ADMIN/'enabled'/a.dataset
    require(not marker.is_symlink(), 'Unexpected enable marker symlink')
    if a.disable:
        registry['datasets'][a.dataset] = dict(enabled=False, reason='Author maintenance; historical results retained')
        write(ADMIN/'registry.json', registry)
        marker.unlink(missing_ok=True)
    else:
        receipt = a.dataset + '-' + str(time.time_ns())
        rel = 'bindings/' + receipt + '.json'
        write(ADMIN/rel, binding)
        write(ADMIN/'bindings'/(receipt+'-acceptance.json'), evidence)
        registry['datasets'][a.dataset] = dict(enabled=True, binding=rel, acceptance=evidence,
            reason='Current source passed short1; short5/full20 are available modes, not prevalidated results')
        atomic(marker, (binding['entry_sha256']+'\n').encode())
        write(ADMIN/'registry.json', registry)
    print(a.dataset + ': ' + ('PENDING_ACCEPTANCE' if a.disable else 'OPEN'))

if __name__ == '__main__':
    main()
