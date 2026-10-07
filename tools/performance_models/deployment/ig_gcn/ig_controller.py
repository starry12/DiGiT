"""Five complete paired rounds; select the maximum observed same-round ratio."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_context
model_context.activate()
import argparse
import csv
import fcntl
import math
import os
import signal
import statistics
import time
from pathlib import Path
from candidates.ig_sage_pair_max5_v1.common import *
import gpu_selection as auto_gpu

ARMS = ('gids', 'digit_full')
LABELS = {'gids': 'GIDS', 'digit_full': 'DiGiT'}


def jobs():
    return [dict(item, model='gcn', smoke=False, profile_mode='host')
            for item in read(HERE / 'protocol.json')['schedule']]


def command(job, output):
    return [PY, '-I', '-B', '-u', '/srv/digit-ae/admin/multimodel_v2/ig_gcn/ig_worker.py',
            '--arm', job['arm'], '--model', 'gcn', '--profile-mode', 'host',
            '--output', str(output / job['mode'] / job['arm']),
            '--binding', str(output / 'inputs.json')]


def summarize(rows):
    expected = [(j['mode'], j['arm'], j['repetition']) for j in jobs()]
    require([(r['mode'], r['arm'], r['repetition']) for r in rows] == expected,
            'Exactly ten ordered complete runs are required')
    for row in rows:
        require(row['accepted'] and row['measured_batches'] == 300 and row['warmup_batches'] == 20,
                'Incomplete or unaccepted run')
        values = row['windows_seconds']
        require(len(values) == 3 and all(math.isfinite(x) and x > 0 for x in values), 'Invalid windows')
        require(math.isfinite(row['seconds']) and
                math.isclose(row['seconds'], sum(values), rel_tol=1e-12, abs_tol=1e-9),
                'Whole-run total differs from windows')
    arms = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        require(len(selected) == 5 and sorted(r['repetition'] for r in selected) == [1, 2, 3, 4, 5],
                'Each arm needs five distinct repetitions')
        values = [r['seconds'] for r in selected]
        arms[arm] = dict(count=5, min_seconds=min(values),
                         median_seconds=statistics.median(values), max_seconds=max(values),
                         all_runs_seconds=values)
    pairs = []
    for number in range(1, 6):
        pair = {r['arm']: r for r in rows if r['repetition'] == number}
        require(set(pair) == set(ARMS), 'Missing system in paired round')
        gids, digit = [pair[arm]['seconds'] for arm in ARMS]
        ratio = gids / digit
        require(math.isfinite(ratio), 'Invalid paired speedup')
        pairs.append(dict(round=number, gids_run=pair['gids']['mode'], digit_run=pair['digit_full']['mode'],
                          gids_seconds=gids, digit_seconds=digit, speedup=ratio,
                          training_time_reduction_percent=100 * (1 - digit / gids)))
    selected = max(pairs, key=lambda p: p['speedup'])  # Raw precision, first round on exact ties.
    return dict(arms=arms, paired_rounds=pairs, selected_round=selected['round'], selected_pair=selected,
                max_observed_speedup=selected['speedup'],
                median_paired_speedup=statistics.median(p['speedup'] for p in pairs),
                min_paired_speedup=min(p['speedup'] for p in pairs),
                selection_rule='maximum_same_round_speedup_out_of_five',
                interference_free_proven=False, stability_proven=False)


def make_row(job, report):
    stages = {}
    for window in report['stage_profile']['windows']:
        if window['phase'] == 'training':
            for name, value in window['spans'].items():
                stages[name] = stages.get(name, 0.) + value['inclusive_seconds']
    return dict(warmup_gate=report['warmup_gate'],mode=job['mode'], arm=job['arm'], repetition=job['repetition'],
                accepted=bool(report['passed'] and report['external_monitor']['strict_monitor_passed']
                              and report['training_telemetry']['diagnostic_complete']),
                seconds=report['training_seconds'],
                windows_seconds=[w['seconds'] for w in report['training']['windows']],
                measured_batches=report['training']['batches'], warmup_batches=report['warmup']['batches'],
                setup_seconds=report['setup_seconds'], warmup_seconds=report['warmup_seconds'],
                cpu_policy=report['affinity']['initial']['policy'],
                ssd_completed_bytes=report['training']['device']['completed_bytes'],
                feature_rows=report['training']['feature_rows'],
                native_feature_seconds=report['training']['feature_seconds'],
                host_stages_seconds=stages)


def review(output):
    setup()
    from candidates.ig_sage_host_telemetry_v1 import controller as parent
    from candidates.ig_sage_host_telemetry_v1.validation import pair_check
    state = read(output / 'status.json')
    require(state['gpu_assignment']==auto_gpu.assignment() and state['transport_sha256']==auto_gpu.transport(),'GPU transport differs')
    launch = read(output / 'launch.json')
    binding = read(output / 'inputs.json')
    require(state['wrapper_sha256'] == launch['wrapper_sha256'] == verify(), 'Wrapper changed')
    require(launch['jobs'] == jobs() and launch['protocol_sha256'] == sha(HERE / 'protocol.json') and
            launch['inputs_sha256'] == sha(output / 'inputs.json'), 'Plan, protocol or binding changed')
    check_inputs(binding)
    require(len(state['workers']) == 10, 'Ten workers required')
    rows, counts, reports = [], {}, {}
    for job, worker in zip(jobs(), state['workers']):
        require(all(worker[k] == v for k, v in job.items()) and worker['command'] == command(job, output),
                'Unexpected worker')
        require(worker['status'] == 'complete' and worker['returncode'] == 0, 'Worker failed')
        prefix = job['mode'] + '_' + job['arm']
        receipt = read(output / (prefix + '_receipt.json'))
        accepted = output / (prefix + '_accepted.json')
        require(receipt['passed'] and receipt['job'] == job and
                receipt['files'] == parent.arm_evidence(output, job), 'Raw evidence changed')
        require(receipt['accepted_sha256'] == sha(accepted) and
                receipt['report_sha256'] == sha(output / job['mode'] / job['arm'] / 'report.json'),
                'Report changed')
        auto_gpu.verify_monitor(output/(job['mode']+'_monitor_'+job['arm'])/'external_gpu')
        report = parent.review_worker(output, worker, state['pid'], binding, auto_gpu.assignment()['index'])
        require(report == read(accepted), 'Native completion checks differ')
        rows.append(make_row(job, report))
        counts[job['mode']] = len(receipt['files'])
        reports[job['mode']] = report
    require(all(a['finished_unix'] <= b['started_unix'] for a, b in zip(state['workers'], state['workers'][1:])),
            'Runs overlapped')
    stats = summarize(rows)
    # Offline metadata comparison only; it never runs an extra GPU workload.
    pairs = {str(i): pair_check({'gids': reports['gids_%d' % i],
                                 'digit_full': reports['digit_%d' % i]}, False) for i in range(1, 6)}
    for pair in stats['paired_rounds']:
        require(pairs[str(pair['round'])]['training_speedup'] == pair['speedup'], 'Paired ratio differs')
    best_pair = pairs[str(stats['selected_round'])]
    require(best_pair['training_speedup'] == stats['max_observed_speedup'], 'Selected pair ratio differs')
    return dict(model_identity=model_context.verify(),gpu_assignment=auto_gpu.assignment(),transport_sha256=auto_gpu.transport(),passed=True, complete=True, wrapper_sha256=verify(), native_parent_sha256=PARENT_SHA,
                rows=rows, statistics=stats, paired_workload_checks=pairs, selected_pair_workload_check=best_pair,
                evidence_files_by_worker=counts, strict_resource_acceptance=True, diagnostic_complete=True,
                limits=read(HERE / 'protocol.json')['limits'])


def emit(result, output):
    write(output / 'summary.json', result)
    stats = result['statistics']
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'system', 'repetition', 'first_100_s', 'middle_100_s', 'last_100_s',
                         'total_300_s', 'selected_pair', 'ssd_completed_bytes'])
        for row in result['rows']:
            writer.writerow([row['mode'], LABELS[row['arm']], row['repetition'],
                             *['%.2f' % x for x in row['windows_seconds']], '%.2f' % row['seconds'],
                             row['repetition'] == stats['selected_round'], row['ssd_completed_bytes']])
    with (output / 'pairs.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'gids_run', 'digit_run', 'gids_seconds', 'digit_seconds',
                         'speedup', 'training_time_reduction_percent', 'selected'])
        for pair in stats['paired_rounds']:
            writer.writerow([pair['round'], pair['gids_run'], pair['digit_run'],
                             *['%.2f' % pair[k] for k in ('gids_seconds', 'digit_seconds', 'speedup',
                                                         'training_time_reduction_percent')],
                             pair['round'] == stats['selected_round']])
    with (output / 'stages.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'system', 'stage', 'seconds', 'nested'])
        for row in result['rows']:
            for stage, seconds in row['host_stages_seconds'].items():
                writer.writerow([row['mode'], LABELS[row['arm']], stage, '%.2f' % seconds, '/' in stage])
    selected = stats['selected_pair']
    lines = ['# IG/GCN：五轮最大观察加速比', '',
             '**五轮最大观察加速比：%.2f×（第%d轮）**。' %
             (stats['max_observed_speedup'], stats['selected_round']), '',
             '选中同一轮的 GIDS %.2f 秒（%s）与 DiGiT %.2f 秒（%s），耗时减少 %.2f%%。' %
             (selected['gids_seconds'], selected['gids_run'], selected['digit_seconds'],
              selected['digit_run'], selected['training_time_reduction_percent']), '',
             '该结果为五轮中最大观察值；全部五轮如下，不能用它代表平均或稳定加速。',
             'GIDS保持默认CPU调度，DiGiT沿用原版CPU2绑定；每次预热20批、计时300批。', '',
             '| 轮次 | GIDS（秒） | DiGiT（秒） | 同轮加速比 | 展示结果 |',
             '| --- | ---: | ---: | ---: | --- |']
    for pair in stats['paired_rounds']:
        lines.append('| %d | %.2f | %.2f | %.2f× | %s |' %
                     (pair['round'], pair['gids_seconds'], pair['digit_seconds'], pair['speedup'],
                      '最大观察值' if pair['round'] == stats['selected_round'] else '保留'))
    lines += ['', '五轮加速比中位数 %.2f×，最小值 %.2f×。按原始精度选择，表中保留两位小数。' %
              (stats['median_paired_speedup'], stats['min_paired_speedup']), '',
              '| Run | First 100 (s) | Middle 100 (s) | Last 100 (s) | All 300 (s) |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        lines.append('| %s | %.2f | %.2f | %.2f | %.2f |' %
                     (row['mode'], *row['windows_seconds'], row['seconds']))
    lines += ['', 'Existing host stages are exported to stages.csv; nested spans overlap their parents.',
              '', *['- ' + value for value in result['limits']]]
    (output / 'README.md').write_text('\n'.join(lines) + '\n')


def execute(output):
    from candidates.ig_sage_host_telemetry_v1 import controller as parent
    from candidates.ig_sage_host_telemetry_v1.inputs import bind
    output.mkdir(parents=True, exist_ok=False)
    state = dict(gpu_assignment=auto_gpu.assignment(),transport_sha256=auto_gpu.transport(),wrapper_sha256=verify(), pid=os.getpid(), workers=[], passed=False,
                 complete=False, started_unix=time.time())
    def save(**values):
        state.update(values, updated_unix=time.time())
        write(output / 'status.json', state)
    save(stage='binding_inputs')
    try:
        binding = bind(output, PARENT_SHA)
        write(output / 'launch.json', dict(wrapper_sha256=state['wrapper_sha256'],
              native_parent_sha256=PARENT_SHA, jobs=jobs(), protocol_sha256=sha(HERE / 'protocol.json'),
              inputs_sha256=sha(output / 'inputs.json')))
        parent.worker_command = command
        rows = []
        for job in jobs():
            report = parent.run_worker(output, job, state, binding, save, auto_gpu.assignment()['index'])
            row = make_row(job, report)
            rows.append(row)
            write(output / 'progress_summary.json', dict(rows=rows, runs_complete=len(rows),
                  per_arm_counts={arm: sum(r['arm'] == arm for r in rows) for arm in ARMS}))
            require(row['accepted'], 'Runtime checks incomplete; retain results and stop without retry')
        result = review(output)
        emit(result, output)
        save(stage='complete', passed=True, complete=True, finished_unix=time.time(),
             max_observed_speedup=result['statistics']['max_observed_speedup'],
             selected_round=result['statistics']['selected_round'])
    except BaseException as exc:
        save(stage='failed', error=type(exc).__name__ + ': ' + str(exc), finished_unix=time.time())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0 and os.environ.get('TMUX') and __debug__, 'Use local sudo and tmux')
    def stop(*_):
        raise KeyboardInterrupt('Five paired rounds interrupted')
    signal.signal(signal.SIGTERM, stop)
    os.environ.update(environment(auto_gpu.assignment()['index']))
    with open('/tmp/digit-pa-bidir-controller.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.output)


if __name__ == '__main__':
    main()
