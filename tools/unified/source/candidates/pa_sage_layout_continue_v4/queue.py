"""Reuse accepted ABBA evidence; run the remaining fourteen frozen v3 points."""
import argparse
import csv
import fcntl
import os
import subprocess
import time
from .common import (ROOT, HERE, SOURCE, OUT, PLAN, INDEX, PY, UNIT, NATIVE, NATIVE_SHA,
                     Path, read, write, sha, require, host, verify)
from .compare import collect
from candidates.pa_sage_layout_native_v2.common import ORDER, GPU_UUID
TMP = Path('/mnt/n0/digit/pa_sage_layout_shared_resume_20260925_v3/.tmp')


def prerequisites():
    previous = {}
    for unit in ('digit-pa-layout-shared-20260925-v1.service',
                 'digit-pa-layout-shared-fast-20260925-v2.service',
                 'digit-pa-layout-shared-resume-20260925-v3.service'):
        text = subprocess.check_output(['/usr/bin/systemctl', 'show', unit,
            '--property=ActiveState,MainPID'], text=True, timeout=30)
        d = dict(line.split('=', 1) for line in text.splitlines() if '=' in line)
        require(d['ActiveState'] in ('inactive', 'failed') and int(d['MainPID']) == 0,
                'Previous service still active: ' + unit)
        previous[unit] = d
    old = read(SOURCE / 'status.json')
    require(old['stage'] == 'failed' and old['error'] == 'RuntimeError: Feature request count differs'
            and not old['completed'] and old['candidate_sha256'] == NATIVE_SHA, 'Unexpected earlier state')
    review = collect()
    require(review['passed'] and review == read(OUT / 'calibration_review.json'),
            'Corrected calibration not passed or evidence changed')
    from candidates.pa_sage_layout_shared_resume_v3.pool import check_receipt
    receipts = {mode: check_receipt(mode) for mode in ('real', 'shared')}
    return dict(previous_services=previous, calibration_review_sha256=sha(OUT / 'calibration_review.json'),
                pool_receipt_sha256={k: sha(v['receipt_path']) for k, v in receipts.items()},
                raw_ssd_writes=False, new_pool_readback=False, new_calibration_workers=False)


def cells():
    index = read(INDEX)
    items = {r['point']['id']: r for r in index['points']}
    plan = {r['id']: r for r in read(PLAN)['points']}
    require(len(index['points']) == len(items) == len(plan) == 15 and
            set(items) == set(plan) == set(ORDER), 'Incomplete grid')
    result = []
    for key in ORDER:
        item = items[key]
        require(sha(item['protocol']) == item['protocol_sha256'], 'Protocol changed')
        p = read(item['protocol'])
        require(p['base_layout'] == plan[key]['filesystem_destination'] and
                str(p['base_layout']).startswith('/mnt/n0/'), 'Wrong large-file destination')
        require(p['epochs'] == 1 and p['evaluation'] == 'disabled' and p['feature_mode'] == 'shared',
                'Wrong performance protocol')
        from candidates.pa_sage_layout_shared_resume_v3.protocol import validate
        validate(p)
        if key == 'g2_r20':
            continue
        require(not p['verification_policy']['independent_native_smoke'] and
                not p['verification_policy']['full_graph_semantics'], 'Wrong remaining-point scope')
        result.append((key, Path(item['protocol'])))
    require(len(result) == 14, 'Need exactly fourteen new points')
    return result


def commands(key, protocol):
    return [
        ('layout_' + key, [PY, '-B', '-u', '-m', NATIVE + '.build', '--plan', str(PLAN),
                         '--point', key, '--execute-filesystem-build']),
        ('overlay_' + key, [PY, '-B', '-u', '-m', NATIVE + '.queue', '--overlay', key]),
        ('budget_' + key, [PY, '-B', '-u', '-m', NATIVE + '.cli', '--protocol', str(protocol),
                          '--budget-output', str(OUT / 'preflight' / (key + '.json'))]),
        ('native_' + key, [PY, '-B', '-u', '-m', NATIVE + '.cli', '--protocol', str(protocol),
                          '--output', str(OUT / 'native' / key), '--execute'])]


def available(required_host=192 * 2**30, required_gpu=0):
    while True:
        values = subprocess.check_output(['nvidia-smi', '-i', '2',
            '--query-gpu=uuid,memory.free,memory.total', '--format=csv,noheader,nounits'],
            text=True, timeout=30).strip().split(',')
        uuid, free, total = values[0].strip(), int(values[1]) * 2**20, int(values[2]) * 2**20
        require(uuid == GPU_UUID and required_gpu <= total, 'Wrong GPU or point exceeds capacity')
        ram = host()
        if ram >= required_host and free >= max(total - 2**30, required_gpu):
            return
        write(OUT / 'resource_wait.json', dict(host_available=ram, host_required=required_host,
              gpu_free=free, gpu_required=max(total - 2**30, required_gpu), updated_unix=time.time()))
        time.sleep(30)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require(os.geteuid() == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'Root / physical GPU 2 required')
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (OUT / 'status.json').exists(), 'Preserve existing run; review before another continuation')
    execution = verify()
    state = dict(schema='digit-shared-pool-grid-continuation-v4', passed=False, complete=False,
        stage='starting', candidate_sha256=execution, native_candidate_sha256=NATIVE_SHA,
        pid=os.getpid(), started_unix=time.time(), steps=[], completed=[], raw_ssd_writes=False)

    def save(stage, **kw):
        state.update(stage=stage, updated_unix=time.time(), **kw)
        write(OUT / 'status.json', state)
        print(stage, kw, flush=True)

    def child(name, command):
        verify()
        save(name)
        folder = OUT / 'logs'
        folder.mkdir(exist_ok=True)
        row = dict(name=name, command=command, started_unix=time.time())
        state['steps'].append(row)
        write(OUT / 'status.json', state)
        with (folder / (name + '.log')).open('x') as log:
            try:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                row['returncode'] = 0
            except subprocess.CalledProcessError as exc:
                row['returncode'] = exc.returncode
                raise
            finally:
                row['finished_unix'] = time.time()
                write(OUT / 'status.json', state)

    try:
        save('checking_reused_calibration')
        write(OUT / 'reuse.json', prerequisites())
        points = cells()
        native = OUT / 'native'
        native.mkdir(exist_ok=True)
        (native / 'g2_r20').symlink_to(SOURCE / 'native/g2_r20', target_is_directory=True)
        state['completed'].append('g2_r20')
        TMP.mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(TMP)
        save('calibration_reused')
        for key, protocol in points:
            save('waiting_layout_' + key)
            available(100 * 2**30)
            steps = commands(key, protocol)
            for name, command in steps[:3]:
                child(name, command)
            budget = read(OUT / 'preflight' / (key + '.json'))
            save('waiting_native_' + key)
            available(budget['host_required_bytes'], budget['required_bytes'])
            child(*steps[3])
            state['completed'].append(key)
            save('point_complete', point=key)
        from candidates.pa_sage_layout_shared_resume_v3.aggregate import collect as aggregate
        result = aggregate(read(INDEX), native)
        result.update(calibration_review_sha256=sha(OUT / 'calibration_review.json'),
                      continuation_candidate_sha256=execution, native_candidate_sha256=NATIVE_SHA)
        write(OUT / 'summary.json', result)
        fields = ['point', 'group_size', 'replica_percent', 'achieved_replica_fraction',
                  'mean_training_epoch_seconds', 'order_excluded_mean_seconds',
                  'speedup_vs_fresh_g2_r20', 'order_excluded_speedup_vs_g2_r20',
                  'layout_address_span_bytes', 'groups', 'padding_rows', 'group_sample_fraction']
        with (OUT / 'summary.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in result['points']:
                row = {k: r[k] for k in fields if k not in ('point', 'group_size', 'replica_percent')}
                row.update(point=r['point']['id'], group_size=r['point']['group_size'], replica_percent=r['point']['replica_percent'])
                writer.writerow(row)
        verify()
        save('complete', passed=True, complete=True, finished_unix=time.time())
    except BaseException as exc:
        save('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
