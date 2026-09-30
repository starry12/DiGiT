#!/usr/bin/python3 -I
"""DiGiT reviewer CLI: start, inspect progress, and display recorded results."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

CONTROL = Path('/srv/digit-ae/admin/selfservice_v2')
NATIVE = Path('/srv/digit-ae/releases/digit_ae_20260923_v3/results/native')
PAIRS = {d.lower() + '-' + m: (d, m) for d in ('PA', 'IG') for m in ('sage', 'gcn', 'gat')}
PAIRS['ig-all'] = ('IG', 'all')
ACTIONS = ('check', 'smoke', 'run')
ARMS = ('gids', 'digit_full')
ACTIVE = ('active', 'activating', 'deactivating', 'reloading')


def actions(pair):
    return ('run',) if pair == 'ig-all' else ACTIONS


def unit(pair, action):
    if pair not in PAIRS or action not in actions(pair):
        raise ValueError('Unsupported experiment')
    return 'digit-ae-' + pair + '-' + action + '.service'


def read(path, required=False):
    path = Path(path)
    try:
        with path.open('rb') as stream:
            raw = stream.read(32 * 1024**2 + 1)
    except FileNotFoundError:
        if required:
            raise ValueError('Missing evidence: ' + str(path))
        return {}
    if len(raw) > 32 * 1024**2:
        raise ValueError('Unexpectedly large JSON: ' + str(path))
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Unreadable JSON: ' + str(path)) from exc
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def native_path(path):
    path = Path(path).resolve()
    path.relative_to(NATIVE.resolve())
    return path


def evidence(folder, name, required=False):
    return read(native_path(folder / name), required)


def service_state(pair, action):
    name = 'digit-ae-selftest.service' if pair == 'selftest' else unit(pair, action)
    try:
        r = subprocess.run(['/usr/bin/systemctl', 'show', name,
                            '--property=ActiveState,SubState,Result'],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {'unit': name, 'ActiveState': 'unknown', 'Result': 'unknown'}
    fields = dict(line.split('=', 1) for line in r.stdout.splitlines() if '=' in line)
    fields['unit'] = name
    if r.returncode:
        fields.update(ActiveState='unknown', Result='unknown')
    return fields


def latest_request(pair, action=None):
    # A busy/rejected request updates only its per-action pointer. Include it so
    # an older successful generic pointer never masks the most recent failure.
    names = [pair + '-' + action] if action else [pair, *[pair + '-' + a for a in (('selftest',) if pair == 'selftest' else actions(pair))]]
    records = [read(CONTROL / 'state/latest' / (name + '.json')) for name in names]
    records = [r for r in records if r]
    for r in records:
        if r.get('pair') != pair or r.get('action') not in (('selftest',) if pair == 'selftest' else actions(pair)):
            raise ValueError('Latest request identity mismatch: ' + pair)
        native_path(r['output'])
    return max(records, key=lambda r: r.get('started_unix', 0)) if records else None


def snapshot(pair, action=None):
    req = latest_request(pair, action)
    out = dict(pair=pair, state='NOT_STARTED', action=action, details=[])
    if not req:
        return out
    folder = native_path(req['output'])
    state = evidence(folder, 'status.json')
    service = service_state(pair, req['action'])
    stage = state.get('stage', 'waiting_for_status')
    if stage in ('failed', 'interrupted'):
        result = stage.upper()
    elif service.get('ActiveState') == 'failed' or (service.get('Result') not in (None, 'success', 'unknown') and service.get('ActiveState') not in ACTIVE):
        result = 'FAILED'
    elif state.get('complete') is True and state.get('passed') is True and stage == 'complete':
        result = 'FINALIZING' if service.get('ActiveState') in ACTIVE else 'PASS'
    elif service.get('ActiveState') in ACTIVE:
        result = 'RUNNING'
    else:
        result = 'INCOMPLETE'
    now = time.time()
    end = state.get('updated_unix', now) if stage in ('complete', 'failed', 'interrupted') else now
    out.update(gpu=state.get('gpu',req.get('gpu')),gpu_uuid=state.get('gpu_uuid',req.get('gpu_uuid')),action=req['action'], state=result, stage=stage, output=str(folder),
               elapsed_seconds=max(0, end - req['started_unix']), service=service,
               error=state.get('error'), completed_pairs=state.get('completed_pairs', []),
               native_acceptance=state.get('native_acceptance', False),
               package_sha256=req.get('package_sha256'))
    pairs = ['ig-sage', 'ig-gcn', 'ig-gat'] if pair == 'ig-all' else [pair]
    for key in pairs:
        base = folder / key if pair == 'ig-all' else folder
        check = evidence(base, 'check/status.json')
        experiment = evidence(base, 'experiment/status.json')
        detail = dict(pair=key, preflight='PASS' if check.get('passed') and check.get('complete') else check.get('stage', 'cached or not yet run'),
                      preflight_seconds=check.get('seconds'), stage=experiment.get('stage', check.get('stage', stage)))
        worker = next((w for w in reversed(experiment.get('workers', [])) if w.get('status') != 'complete'), None)
        if worker and stage not in ('complete', 'failed', 'interrupted'):
            arm, mode = worker.get('arm'), worker.get('mode')
            if arm in ARMS and mode in ('smoke', 'full', 'benchmark'):
                progress = evidence(base, 'experiment/' + mode + '/' + arm + '/progress.json')
                detail.update(arm=arm, phase=mode, worker_stage=progress.get('stage', worker.get('status')),
                              epoch=progress.get('epoch'), batch=progress.get('updates', progress.get('batch')),
                              progress_age_seconds=max(0, now - progress['time_unix']) if progress.get('time_unix') else None)
        out['details'].append(detail)
    return out


def require(ok, message):
    if not ok:
        raise ValueError(message)


def closed_result(base, pair, action, package):
    dataset, model = PAIRS[pair]
    sub = base / ('check' if action == 'check' else 'experiment')
    state = evidence(sub, 'status.json', True)
    receipt = evidence(sub, 'artifact_invocation.json', True)
    require(state.get('complete') is True and state.get('passed') is True, 'Result is not closed and accepted')
    require(receipt.get('exit_code') == 0 and receipt.get('dataset') == dataset and receipt.get('model') == model,
            'Invocation receipt failed or names a different experiment')
    require(receipt.get('package_sha256') == package and receipt.get('action') == ('representative' if action == 'run' else action),
            'Invocation package/action differs')
    result = dict(pair=pair, kind=action, strict_resource_acceptance=None)
    if action == 'check':
        result.update(scope='Filesystem, receipts and admission only; no native training.',
                      preflight_seconds=state.get('seconds'), rows=[])
        return result
    if action == 'smoke':
        summary = evidence(sub, 'smoke_summary.json', True)
        require(summary.get('passed') is True, 'Paired smoke summary is not accepted')
        rows = []
        for arm in ARMS:
            name = 'smoke_' + arm + '_accepted.json'
            report = evidence(sub, name, True)
            digest = hashlib.sha256(native_path(sub / name).read_bytes()).hexdigest()
            require(summary.get('report_sha256', {}).get(arm) == digest, 'Accepted smoke report hash differs: ' + arm)
            require(report.get('passed') is True and report.get('arm') == arm, 'Smoke arm was not accepted: ' + arm)
            monitor = report.get('external_monitor', {})
            lifecycle = report.get('evaluation_lifecycle', {})
            rows.append(dict(arm=arm, updates=report.get('updates'), validation_calls=lifecycle.get('validation_calls'),
                             test_calls=lifecycle.get('test_calls'), monitor=monitor,
                             observed_peak_gpu_bytes=monitor.get('observed_peak_device_used_bytes'),
                             peak_host_rss_kib=report.get('peak_host_rss_kib')))
        result.update(scope='Native smoke only. No full performance or final accuracy measurement.', rows=rows)
    else:
        summary = evidence(sub, 'submission_summary/summary.json', True)
        require(summary.get('passed') is True and summary.get('complete') is True, 'Full summary is not accepted')
        rows = summary.get('rows', [])
        require(len(rows) == 2 and {r.get('arm') for r in rows} == set(ARMS), 'Summary must contain exactly GIDS and DiGiT')
        require(all(r.get('dataset') == dataset and r.get('model') == model for r in rows), 'Summary dataset/model differs')
        require('author_accepted' not in summary.get('provenance', ''), 'Historical reference is not a new run')
        if dataset == 'PA':
            require(summary.get('training_accuracy_io_passed') is True and all(r.get('epochs') == 20 and r.get('test_accuracy') is not None for r in rows), 'PA full results require 20 epochs and final test for both arms')
        result.update(rows=sorted(rows, key=lambda r: ARMS.index(r['arm'])), ratios=summary.get('ratios', {}),
                      scope='20-epoch training, validation and final test.' if dataset == 'PA' else 'Fixed mini-batch windows only; no accuracy or epoch-time claim.',
                      provenance=summary.get('provenance'), limitations=summary.get('limitations', []))
    require(isinstance(summary.get('strict_resource_acceptance'), bool), 'Missing strict monitoring qualification')
    require(all(isinstance(r.get('monitor', {}).get('strict_monitor_passed'), bool) for r in result['rows']), 'Missing arm monitoring qualification')
    require(summary['strict_resource_acceptance'] == all(r['monitor']['strict_monitor_passed'] for r in result['rows']), 'Monitoring qualifications disagree')
    result.update(strict_resource_acceptance=summary.get('strict_resource_acceptance'),
                  qualification=summary.get('qualification'))
    return result


def positive_number(value, zero=False):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def completed_epoch_metrics(epochs):
    """Only closed epoch records, with ratios of sums, not means of rates."""
    if not epochs:
        return dict(epochs=0, training_seconds=None, mean_training_epoch_seconds=None,
                    validation_seconds=None, io={}, io_complete=False)
    seconds = sum(e['train_seconds'] for e in epochs)
    result = dict(epochs=len(epochs), training_seconds=seconds,
                  mean_training_epoch_seconds=seconds / len(epochs),
                  validation_seconds=sum(e['validation']['seconds'] for e in epochs),
                  io={}, io_complete=False)
    counters = dict(ssd_useful_bytes=0, ssd_completed_bytes=0, ssd_active_seconds=0,
                    logical_feature_bytes=0, feature_seconds=0)
    try:
        for e in epochs:
            t = e['training']; u = t['useful_io']; d = t['device']; f = t['feature']
            require(u['schema'] == 'digit-useful-io-v1' and u['enabled'] and u['reconciled'] and d['enabled'] and d['reconciled'], 'Unreconciled live I/O')
            values = [u[k] for k in ('ssd_useful_bytes', 'ssd_fill_bytes', 'gpu_feature_bytes', 'gpu_feature_rows')]
            values += [d[k] for k in ('completed_bytes', 'primary_bytes', 'replay_bytes', 'active_ns', 'submitted_commands', 'completed_commands')]
            values += [f['cpu'], f['gpu_ssd']]
            require(all(type(v) is int and v >= 0 for v in values), 'Invalid live I/O counters')
            width = t['feature_row_bytes']
            require(width in (512, 4096) and u['gpu_feature_rows'] == f['gpu_ssd'] and u['gpu_feature_bytes'] == f['gpu_ssd'] * width, 'Live feature accounting differs')
            require(d['completed_bytes'] == d['primary_bytes'] + d['replay_bytes'] and d['submitted_commands'] == d['completed_commands'], 'Unfinished live I/O region')
            require(u['ssd_fill_bytes'] == d['primary_bytes'] and u['ssd_useful_bytes'] <= min(u['ssd_fill_bytes'], u['gpu_feature_bytes']), 'Invalid useful payload')
            require(positive_number(t['feature_seconds'], zero=True), 'Invalid feature duration')
            counters['ssd_useful_bytes'] += u['ssd_useful_bytes']
            counters['ssd_completed_bytes'] += d['completed_bytes']
            counters['ssd_active_seconds'] += d['active_ns'] / 1e9
            counters['logical_feature_bytes'] += (f['cpu'] + f['gpu_ssd']) * width
            counters['feature_seconds'] += t['feature_seconds']
        for key, num, den in (('ssd_useful_gbps', 'ssd_useful_bytes', 'ssd_active_seconds'),
                              ('ssd_physical_gbps', 'ssd_completed_bytes', 'ssd_active_seconds'),
                              ('effective_feature_gbps', 'logical_feature_bytes', 'feature_seconds')):
            counters[key] = counters[num] / counters[den] / 1e9 if counters[den] > 0 else None
        result.update(io=counters, io_complete=True)
    except (KeyError, TypeError, ValueError):
        # Timing remains useful if an old/incomplete epoch lacks I/O counters.
        # Never silently sum only a subset of its I/O regions.
        pass
    return result


def live_performance(base, pair):
    experiment = base / 'experiment'
    state = evidence(experiment, 'status.json')
    dataset, model = PAIRS[pair]
    require(state.get('dataset') == dataset and state.get('model') == model and state.get('mode') == 'representative',
            'Live experiment identity/mode is not available or differs')
    epochs_by_arm = {}; warnings = []
    for arm in ARMS:
        rows = []
        for index in range(1, 21):
            name = 'full/{}/epoch_{:02d}.json'.format(arm, index)
            try:
                e = evidence(experiment, name)
                if not e:
                    break
                require(e.get('epoch') == index and type(e.get('updates')) is int and e['updates'] > 0 and
                        positive_number(e.get('train_seconds')) and positive_number(e.get('validation', {}).get('seconds'), zero=True),
                        'Incomplete epoch timing')
            except (ValueError, OSError, TypeError, KeyError) as exc:
                warnings.append('Stopped before ' + name + ': ' + str(exc)); break
            rows.append(e)
        epochs_by_arm[arm] = rows
    common = min(len(rows) for rows in epochs_by_arm.values())
    # A speedup is only meaningful for matching epoch IDs AND update counts.
    for i in range(common):
        if epochs_by_arm['gids'][i]['updates'] != epochs_by_arm['digit_full'][i]['updates']:
            common = i
            warnings.append('Update counts differ; comparison stops before epoch ' + str(i + 1)); break
    rows = []
    for arm in ARMS:
        observed = epochs_by_arm[arm]
        selected = observed[:common] if common else observed
        metrics = completed_epoch_metrics(selected)
        worker = next((w for w in state.get('workers', []) if w.get('mode') == 'full' and w.get('arm') == arm), {})
        metrics.update(arm=arm, published_epochs=len(observed), planned_epochs=20,
                       worker_state=worker.get('status', 'not started'), included_epoch_ids=[e['epoch'] for e in selected])
        if selected and not metrics['io_complete']:
            warnings.append(arm + ': I/O unavailable for this complete comparison scope; displayed as N/A')
        rows.append(metrics)
    a, b = rows
    return dict(pair=pair, provisional=True, native_acceptance=False, common_epochs=common,
                scope=('First {} completed epochs in BOTH arms'.format(common) if common else
                       'Per-arm completed epochs; unmatched workloads, no speedup'),
                rows=rows, training_speedup=a['training_seconds'] / b['training_seconds'] if common else None,
                warnings=warnings, final_test_shown=False,
                note='Only contiguous published epochs, after their validation completes; current epoch excluded. Final pair acceptance is pending.')


def render_live(value):
    print('\n  LIVE PERFORMANCE / PROVISIONAL - final acceptance pending')
    print('  ' + value['scope'])
    for r in value['rows']:
        print('  {}: {} published epochs / {}; worker {}'.format('GIDS' if r['arm'] == 'gids' else 'DiGiT',
              r['published_epochs'], r['planned_epochs'], r['worker_state']))
    specs = [(label, lambda r, k=key: r.get(k), 1, decimals) for label, key, decimals in (
        ('Epochs included in this table', 'epochs', 0), ('Training (s)', 'training_seconds', 2),
        ('Mean training epoch (s)', 'mean_training_epoch_seconds', 2), ('Validation total (s)', 'validation_seconds', 2))]
    specs += [(label, lambda r, k=key: r['io'].get(k), scale, 3) for label, key, scale in (
        ('Useful SSD / active time (GB/s)', 'ssd_useful_gbps', 1), ('Physical SSD / active time (GB/s)', 'ssd_physical_gbps', 1),
        ('Logical features / feature time (GB/s)', 'effective_feature_gbps', 1), ('Physical SSD bytes (GiB)', 'ssd_completed_bytes', 1 / 2**30))]
    table(value['rows'], specs)
    if value['training_speedup'] is not None:
        print('  Provisional training speedup (GIDS/DiGiT, same epochs): ' + number(value['training_speedup'], decimals=4) + 'x')
    else:
        print('  Provisional speedup: N/A (no matching completed epochs yet)')
    print('  ' + value['note'])
    print('  I/O uses summed counters/time; GB/s is decimal. Final test and final speedup appear after pair acceptance.')
    for warning in value['warnings']:
        print('  Note: ' + warning)


def results(pair, action=None):
    view = snapshot(pair, action)
    view['results'] = []
    if view['state'] != 'PASS':
        if view['state'] in ('RUNNING', 'FINALIZING') and view['action'] == 'run' and pair.startswith('pa-'):
            try:
                view['live_performance'] = live_performance(native_path(view['output']), pair)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                view['live_performance_note'] = str(exc)
        return view
    base = native_path(view['output'])
    try:
        for key in (['ig-sage', 'ig-gcn', 'ig-gat'] if pair == 'ig-all' else [pair]):
            result = closed_result(base / key if pair == 'ig-all' else base, key, view['action'], view['package_sha256'])
            view['results'].append(result)
        if any(r['strict_resource_acceptance'] is False for r in view['results']):
            view['state'] = 'PASS_WITH_MONITORING_GAPS'
    except (ValueError, OSError, KeyError, TypeError) as exc:
        view.update(state='INCOMPLETE_EVIDENCE', error=str(exc), results=[])
    return view


def number(value, scale=1, decimals=2):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        return 'N/A'
    return ('{:.%df}' % decimals).format(value * scale)


def duration(value):
    minutes, seconds = divmod(round(value), 60)
    hours, minutes = divmod(minutes, 60)
    return ('%dh ' % hours if hours else '') + '%dm %02ds' % (minutes, seconds)


def table(rows, specs):
    print('  {:<39} {:>13} {:>13}'.format('Metric', 'GIDS', 'DiGiT'))
    for label, getter, scale, decimals in specs:
        print('  {:<39} {:>13} {:>13}'.format(label, *[number(getter(r), scale, decimals) for r in rows]))


def render(view, show_results=False):
    dataset, model = PAIRS.get(view['pair'], ('CPU', 'selftest'))
    print('{} / {} | {} | {}'.format(dataset, {'sage': 'GraphSAGE', 'gcn': 'GCN', 'gat': 'GAT'}.get(model, model), view.get('action') or '-', view['state']))
    if not view.get('output'):
        print('  No matching request has been recorded.'); return
    print('  Workflow elapsed: ' + duration(view['elapsed_seconds']))
    svc = view['service']
    print('  Service: {} / {} (result: {})'.format(svc.get('ActiveState', 'unknown'), svc.get('SubState', 'unknown'), svc.get('Result', 'unknown')))
    if view.get('gpu') is not None:print('  GPU: '+str(view['gpu']))
    print('  Phase: ' + view['stage'])
    for d in view['details']:
        print('  {}: preflight={}{}; stage={}'.format(d['pair'], d['preflight'],
              ' (' + duration(d['preflight_seconds']) + ')' if d.get('preflight_seconds') is not None else '', d['stage']))
        if d.get('arm'):
            print('    {} / {}: {}'.format('GIDS' if d['arm'] == 'gids' else 'DiGiT', d['phase'], d['worker_stage']))
            fields = ['{}={}'.format(k, d[k]) for k in ('epoch', 'batch') if d.get(k) is not None]
            if d.get('progress_age_seconds') is not None:
                fields.append('last progress ' + duration(d['progress_age_seconds']) + ' ago')
            if fields: print('    ' + ', '.join(fields))
    if view.get('error'):
        print('  Error: ' + view['error'])
    for result in view.get('results', []):
        print('\n  ' + result['pair'] + ': ' + result['scope'])
        if result['kind'] == 'check': continue
        print('  Strict monitoring: ' + {True: 'PASS', False: 'NOT PASSED', None: 'UNKNOWN'}[result['strict_resource_acceptance']])
        rows = result['rows']
        if result['kind'] == 'smoke':
            specs = [(label, lambda r, k=key: r.get(k), 1, 0) for label, key in (
                ('Training updates', 'updates'), ('Small validation calls', 'validation_calls'), ('Final test calls', 'test_calls'))]
        else:
            specs = [(label, lambda r, k=key: r.get(k), scale, 2) for label, key, scale in (
                ('Training / measured window (s)', 'training_seconds', 1), ('Validation total (s)', 'validation_seconds', 1),
                ('Final test (s)', 'test_seconds', 1), ('Train + valid + test (s)', 'online_seconds', 1),
                ('Worker incl. setup/checks (s)', 'worker_seconds', 1), ('Final test accuracy (%)', 'test_accuracy', 100))]
            phase = 'training'
            specs += [(label, lambda r, k=key: r.get('io', {}).get(phase, {}).get(k), scale, 3) for label, key, scale in (
                ('Useful SSD / active time (GB/s)', 'ssd_useful_gbps', 1), ('Physical SSD / active time (GB/s)', 'ssd_physical_gbps', 1),
                ('Logical features / feature time (GB/s)', 'effective_feature_gbps', 1), ('Physical SSD bytes (GiB)', 'ssd_completed_bytes', 1 / 2**30))]
        specs += [('Monitor samples', lambda r: r.get('monitor', {}).get('sample_count'), 1, 0),
                  ('Monitor timeouts', lambda r: r.get('monitor', {}).get('timeout_count'), 1, 0),
                  ('Observed GPU peak (GiB)', lambda r: r.get('observed_peak_gpu_bytes'), 1 / 2**30, 2)]
        if result['kind'] == 'smoke':
            specs += [('Host RSS peak (GiB)', lambda r: r.get('peak_host_rss_kib'), 1 / 2**20, 2)]
        table(rows, specs)
        if result['kind'] == 'smoke':
            print('  GPU peaks are sampled lower bounds; smoke timings are not full performance evidence.')
        if result['kind'] == 'run':
            ratio = result.get('ratios', {}).get('training_speedup')
            print('  Training speedup (GIDS/DiGiT): ' + number(ratio, decimals=4) + 'x')
            print('  I/O shown for training/measured windows; GB/s is decimal. GPU peaks are sampled lower bounds.')
            print('  Single seed; no statistical accuracy claim. Provenance: ' + str(result.get('provenance')))
    if view.get('live_performance'):
        render_live(view['live_performance'])
    if view.get('live_performance_note'):
        print('  Live performance not yet available: ' + view['live_performance_note'])
    if show_results and not view.get('results'):
        print('  Final summary pending; live values above are provisional.' if view.get('live_performance') else
              '  No accepted final summary displayed; inspect status/logs. Earlier successes are not substituted.')
    print('\n  Output: ' + view['output'])
    if view.get('results'):
        suffix = 'check/status.json' if view['action'] == 'check' else ('experiment/smoke_summary.json' if view['action'] == 'smoke' else 'experiment/submission_summary/summary.json')
        print('  Evidence: ' + ('<pair>/' if view['pair'] == 'ig-all' else '') + suffix)


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=(*ACTIONS, 'status', 'results', 'logs', 'stop', 'selftest'))
    p.add_argument('dataset', nargs='?', choices=('PA', 'IG'))
    p.add_argument('model', nargs='?', choices=('sage', 'gcn', 'gat', 'all'))
    p.add_argument('--action', dest='request_action', choices=ACTIONS, help='Inspect latest check, smoke or run explicitly')
    p.add_argument('--json', action='store_true', help='Machine-readable status/results')
    a = p.parse_args(argv)
    if bool(a.dataset) != bool(a.model): p.error('Provide both dataset and model')
    pair = a.dataset.lower() + '-' + a.model if a.dataset else None
    if pair and pair not in PAIRS: p.error('Unsupported dataset/model')
    if a.action in (*ACTIONS, 'logs', 'results') and not pair: p.error('Dataset and model are required')
    if pair == 'ig-all' and (a.action in ('check', 'smoke') or a.request_action in ('check', 'smoke')): p.error('IG all only supports the run queue')
    if a.action == 'selftest' and pair: p.error('selftest takes no dataset/model')
    if a.request_action and (not pair or a.action not in ('status', 'results', 'logs')): p.error('--action requires a pair and status/results/logs')
    if a.json and a.action not in ('status', 'results'): p.error('--json is only for status/results')
    return a, pair


def main(argv=None):
    a, pair = parse(sys.argv[1:] if argv is None else argv)
    if a.action in ACTIONS or a.action == 'selftest':
        name = unit(pair, a.action) if pair else 'digit-ae-selftest.service'
        subprocess.run(['/usr/bin/sudo', '-n', '/usr/bin/systemctl', '--no-block', 'start', name], check=True)
        print('Start requested: ' + name)
        if pair:
            suffix = ' '.join(PAIRS[pair])
            print('Progress: digit-ae status ' + suffix + '\nResults:  digit-ae results ' + suffix + '\nLogs:     digit-ae logs ' + suffix)
        print('Runs in background across SSH disconnects. Busy device/lock requests fail; they are not queued.')
        return 0
    if a.action == 'stop':
        names = [unit(k, v) for k in ([pair] if pair else PAIRS) for v in actions(k)]
        if pair is None: names.append('digit-ae-selftest.service')
        for name in names:
            subprocess.run(['/usr/bin/sudo', '-n', '/usr/bin/systemctl', '--no-block', 'stop', name], check=True)
        print('Stop requested. Wait for inactive units and released workers; interrupted output is retained.')
        return 0
    if a.action == 'logs':
        req = latest_request(pair, a.request_action)
        if not req: print('No matching run has been started for ' + pair); return 0
        folder = native_path(req['output'])
        print('Output: ' + str(folder), flush=True)
        for name in ('launcher.log', 'status.json'):
            path = native_path(folder / name)
            if path.is_file():
                print('\n' + name + ' (last 60 lines):', flush=True)
                subprocess.run(['/usr/bin/tail', '-n', '60', '--', str(path)], check=True)
        return 0
    keys = [pair] if pair else [*PAIRS, 'selftest']
    views = [(results if a.action == 'results' else snapshot)(key, a.request_action) for key in keys]
    if a.json:
        print(json.dumps(views[0] if pair else views, indent=2))
    else:
        for view in views:
            render(view, a.action == 'results'); print()
    if a.action == 'results':
        return 0 if views[0]['state'] in ('PASS', 'PASS_WITH_MONITORING_GAPS') else (1 if views[0]['state'] in ('FAILED', 'INTERRUPTED', 'INCOMPLETE_EVIDENCE') else 3)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
