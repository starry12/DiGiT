import datetime
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

def snapshot(folder):
    v = dict(updated_unix=time.time())
    if (folder/'status.json').exists():
        v['entry'] = read(folder/'status.json')
    if (folder/'backend.json').exists():
        backend = read(folder/'backend.json')
        v['backend'] = backend
        p = Path(backend['output'])
        require(inside(p, ROOT/'results'), 'Unexpected backend output')
        for name in ('status.json', 'experiment/status.json'):
            if (p/name).is_file():
                v['backend_status'] = read(p/name)
    return v

def main():
    require(os.geteuid() == 0 and len(sys.argv) == 3, 'Fixed root service required')
    dataset, mode = sys.argv[1:]
    unit(dataset, mode)
    invocation = os.environ.get('INVOCATION_ID', '')
    require(len(invocation) == 32 and all(c in '0123456789abcdef' for c in invocation), 'Missing systemd invocation')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + invocation[:12]
    folder = OUT/dataset/mode/'runs'/stamp
    folder.mkdir(parents=True, exist_ok=False)
    write(OUT/dataset/mode/'latest.json', dict(output=str(folder), invocation_id=invocation))
    child = None
    def stop(signum, frame):
        raise InterruptedError('Reviewer stop requested')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        write(folder/'progress.json', dict(stage='waiting_for_unified_service', updated_unix=time.time()))
        with (ADMIN/'serial.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(5)
            row = read(ADMIN/'registry.json')['datasets'][dataset]
            require(row['enabled'], 'Dataset is pending acceptance')
            binding = read(ADMIN/row['binding'])
            verify_binding(binding)
            work = ROOT/'results/unified_experiments_20261011_v2'/dataset/'runs'/('ae-'+stamp)
            work.mkdir(parents=True, exist_ok=False)
            write(folder/'launch.json', dict(dataset=dataset, mode=mode, backend_output=str(work),
                invocation_id=invocation, entry_sha256=binding['entry_sha256'], acceptance=row['acceptance']))
            cmd = [PYTHON, '-B', '-u', '-m', 'candidates.unified_experiments_v2.run',
                   '--dataset', dataset, '--mode', mode, '--output', str(work)]
            with (folder/'controller.log').open('x') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    try:
                        write(folder/'progress.json', snapshot(work))
                    except (OSError, ValueError):
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode
            require(code == 0, 'Unified backend failed with exit ' + str(code))
            verify_binding(binding)
            state = read(work/'status.json')
            require(state.get('stage') == 'complete' and state.get('passed') and state.get('complete'), 'Incomplete backend')
            require(state['source_sha256'] == binding['entry_sha256'], 'Wrong backend source')
            result = read(work/'summary.json')
            validate_result(result, dataset, mode)
            for name in ('summary.json', 'summary.csv', 'config.json', 'status.json', 'backend.json'):
                atomic(folder/name, (work/name).read_bytes())
            write(folder/'progress.json', snapshot(work))
            write(folder/'completion.json', dict(passed=True, returncode=code, invocation_id=invocation,
                entry_sha256=binding['entry_sha256'], summary_sha256=sha(folder/'summary.json'), finished_unix=time.time()))
    except BaseException as e:
        write(folder/'error.json', dict(error=repr(e), finished_unix=time.time()))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

if __name__ == '__main__':
    main()
