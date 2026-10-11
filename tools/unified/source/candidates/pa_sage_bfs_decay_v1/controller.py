"""Wait for reviewer queues, gate all smoke workers, then run six factorial arms."""
import argparse
import contextlib
import fcntl
import signal
import subprocess
import traceback
from .common import *

LOCKS = ('/run/digit-ae-selfservice/exclusive.lock',
         '/tmp/digit-pa-bidir-controller.lock', '/tmp/digit-pa-sage-libnvm0.lock')


def active_ae():
    result = subprocess.run(['systemctl', 'list-units', 'digit-ae*', '--all',
                             '--no-pager', '--plain', '--no-legend'],
                            capture_output=True, text=True, timeout=15, check=True)
    active = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[2] in ('active', 'activating', 'deactivating') and fields[3] != 'exited':
            active.append(fields[0])
    return active + reviewer_chains()


def reviewer_chains():
    # Wait for the reviewer's whole queue, including sleeps between UKL and CL.
    result=subprocess.run(['ps','-u','atc27_ae','-o','pid=,args='],
                          capture_output=True,text=True,timeout=15)
    require(result.returncode in (0,1),'Cannot inspect reviewer queue')
    return parse_reviewer_chains(result.stdout)


def parse_reviewer_chains(text):
    import re
    active=[]
    for line in text.splitlines():
        parts=line.split()
        if len(parts)>1 and any(re.fullmatch(r'/home/atc27_ae/ae_chain[0-9]*\.sh',arg) for arg in parts[1:]):
            active.append('reviewer_queue_pid_'+parts[0])
    return active


@contextlib.contextmanager
def exclusion(save):
    while True:
        stack = contextlib.ExitStack()
        try:
            active = active_ae()
            if active:
                raise BlockingIOError('Active reviewer units: ' + ', '.join(active))
            handles = []
            for path in LOCKS:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                stack.callback(os.close, fd)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handles.append(fd)
            active = active_ae()
            if active:
                raise BlockingIOError('Reviewer started while admitting: ' + ', '.join(active))
        except BlockingIOError as error:
            stack.close()
            save(stage='waiting_for_reviewers',reason=str(error))
            time.sleep(cfg()["poll_seconds"])
            continue
        except BaseException:
            stack.close()
            raise
        try:
            yield handles[-1]
        finally:
            stack.close()
        return


def gpu_admission():
    result = subprocess.run(['nvidia-smi', '-i', GPU,
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15)
    uuid, used, util = [v.strip() for v in result.stdout.strip().split(',')]
    require(uuid == GPU_UUID and int(used) < 1024 and int(util) == 0, 'GPU 2 occupied or identity changed')
    return dict(uuid=uuid, used_mib=int(used), utilization=int(util), time_unix=time.time())


def run_worker(output, mode, arm, binding, save, ssd_fd):
    from ae.pa_sage.monitor_control import ExternalMonitor
    from submission.v2.review import assess_monitor
    from .validation import report_check
    from .worker import check_inputs
    check_inputs(binding)
    write(output / (mode + '_' + arm + '_gpu.json'), gpu_admission())
    folder = output / mode / arm
    folder.parent.mkdir(exist_ok=True)
    mon = output / (mode + '_monitor_' + arm)
    mon.mkdir()
    monitor = ExternalMonitor(mon / 'external_gpu')
    child = None
    released = False
    worker = dict(arm=arm, mode=mode, started_unix=time.time(), status='starting')
    try:
        ready = monitor.start()
        cmd = [PY, '-B', '-u', '-m', MODULE + '.worker', '--arm', arm,
               '--output', str(folder), '--binding', str(output / 'inputs.json'),
               '--lock-fd', str(ssd_fd)] + (['--smoke'] if mode == 'smoke' else [])
        worker['command'] = cmd
        with (output / (mode + '_' + arm + '.log')).open('x') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(ssd_fd,))
            worker.update(pid=child.pid, status='running', started_unix=time.time())
            save(stage=mode + '_' + arm, worker=worker)
            teardown_deadline = None
            while child.poll() is None:
                if (folder / 'worker_ready.json').exists() and not released:
                    require(read(folder / 'worker_ready.json')['passed'], 'Worker not ready')
                    mon_code = monitor.stop()
                    monitor = None
                    write(folder / 'release_worker.json',dict(monitor_stopped=True,monitor_returncode=mon_code))
                    released = True
                    teardown_deadline = time.monotonic() + 180
                if teardown_deadline:
                    require(time.monotonic() < teardown_deadline, 'Main worker teardown timed out')
                time.sleep(.5)
            code = child.wait()
            child = None
        worker.update(returncode=code, finished_unix=time.time(), status='complete' if code == 0 else 'failed')
        write(output / (mode + '_' + arm + '_exit.json'), worker)
        require(code == 0 and released and mon_code == 0, 'Main worker failed: ' + arm)
        path = folder / 'report.json'
        require(sha(path) == read(folder / 'worker_ready.json')['report_sha256'], 'Report identity differs')
        r = read(path)
        root = mon / 'external_gpu'
        records = [json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()]
        r['external_monitor'] = assess_monitor(read(root / 'summary.json'), records, ready, worker,
            os.getpid(), read(folder / 'resources.json'), mon_code, int(GPU))
        require(r['external_monitor']['strict_monitor_passed'], 'Resource monitor incomplete: ' + mode)
        report_check(r, arm, mode == 'smoke', binding, folder)
        require(r['input_binding_sha256'] == sha(output / 'inputs.json'), 'Wrong inputs')
        require(r['peak_host_rss_kib'] * 1024 <= r['admission']['host_required_bytes'], 'Host budget exceeded')
        for e in r['epochs']:
            require(read(folder / ('epoch_%02d.json' % e['epoch'])) == e, 'Epoch receipt changed')
        check_inputs(binding)
        accepted = output / (mode + '_' + arm + '_accepted.json')
        write(accepted, r)
        write(output / (mode + '_' + arm + '_receipt.json'),
              dict(passed=True, report_sha256=sha(path), accepted_sha256=sha(accepted), worker=worker))
        return r
    finally:
        if monitor is not None:
            monitor.stop()
        # Only this controller's own child may be stopped on failure.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def run(output):
    require(os.geteuid() == 0 and __debug__, 'Local sudo required, assertions enabled')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Expected physical GPU 2')
    queue_fd=os.open('/tmp/digit-pa-bfs-decay-queue.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    fcntl.flock(queue_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)  # retained until controller exits
    output.mkdir(parents=True, exist_ok=False)
    state = dict(passed=False, complete=False, pid=os.getpid(), started_unix=time.time(),
                 scope=cfg()['scope'], completed_stages=[], output=str(output))
    def save(**kw):
        if kw.get('stage') != 'waiting_for_reviewers':
            state.pop('reason', None)
        state.update(kw, updated_unix=time.time())
        write(output / 'status.json',state)
        write(OUT / 'latest.json',state)
        print(json.dumps(state), flush=True)
    def stopped(signum, frame):
        raise KeyboardInterrupt('Main controller signaled')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    save(stage='queued')
    try:
        with exclusion(save) as ssd_fd:
            save(stage='verifying_sources')
            execution = verify()
            require(read(OUT / 'cpu_checks.json')['passed'], 'CPU checks missing')
            require(read(OUT / 'cpu_checks.json')['source_sha256'] == execution, 'Stale CPU checks')
            write(output / 'gpu_admission.json', gpu_admission())
            from .prepare import prepare
            save(stage='binding_inputs_and_preparing_bfs')
            binding = prepare(output, execution)
            from .validation import matrix_check
            for mode in ('smoke','full'):
                reports = {}
                for arm in cfg()['ablation_order']:
                    reports[arm] = run_worker(output,mode,arm,binding,save,ssd_fd)
                    state['completed_stages'].append(mode + '_' + arm)
                    save(stage=mode + '_' + arm + '_accepted')
                result = matrix_check(reports, mode == 'smoke')
                result['bfs_preparation'] = {k:v for k,v in binding['bfs_preparation'].items() if k != 'orders'}
                for arm, row in result['rows'].items():
                    prep = binding['bfs_preparation']['per_arm_order_seconds'][arm]
                    if cfg()['arms'][arm]['bfs_enabled']:
                        prep += binding['bfs_preparation']['canonical_seconds']
                    row['order_preparation_seconds'] = prep
                    row['order_preparation_plus_training_seconds'] = prep + row['training_seconds']
                result['strict_resource_acceptance'] = all(r['external_monitor']['strict_monitor_passed'] for r in reports.values())
                require(result['strict_resource_acceptance'], 'Incomplete resource evidence')
                result['report_sha256'] = {arm:sha(output / (mode + '_' + arm + '_accepted.json')) for arm in reports}
                write(output / (mode + '_summary.json'), result)
            require(verify() == execution, 'Main experiment source changed during run')
            save(stage='complete',passed=True,complete=True,summary_sha256=sha(output/'full_summary.json'),finished_unix=time.time())
    except BaseException as error:
        save(stage='failed',error=str(error),finished_unix=time.time())
        write(output / 'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    run(a.output.resolve())


if __name__ == '__main__':
    main()
