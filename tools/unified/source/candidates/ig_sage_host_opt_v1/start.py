"""DiGiT IG/SAGE affinity and shared-frontier-index optimization, original 20+300 roots."""
import argparse
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/ig_sage_host_opt_20260927_v1'
sys.path.insert(0, str(ROOT))
from candidates.ig_sage_host_opt_v1.common import (check_inputs, environment, host,
    read, require, sha, verify, write)
from candidates.ig_sage_host_opt_v1.common import verify as verify_entry

PY = '/home/embed/miniconda3/envs/gids/bin/python'
SCRIPT = Path(__file__).resolve()
GPU_UUID = 'GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'
PARENT = '99d29f94b8724f2dc95f1e22c394e8c3dd6e2b65d0547f6473a103234b81f5ee'
ENTRY = 'c0f19fa14713d625076a3e2136e1c0b1b00ee5ff9002d3f931c72ce524bde563'


def identities():
    candidate = verify()
    pre = ROOT / 'results/ig_perf_preflight_20260922_v5/check'
    state = read(pre / 'status.json')
    require(state['passed'] and state['complete'] and
            state['candidate_sha256'] == PARENT and
            state['entry_manifest_sha256'] == ENTRY, 'Prior preflight invalid')
    for name, digest in state['evidence_sha256'].items():
        require(sha(pre / name) == digest, 'Prior evidence changed: ' + name)
    binding = read(OUT / 'inputs.json')
    require(binding['candidate_sha256'] == candidate, 'New input binding differs')
    check_inputs(binding)
    return dict(candidate_sha256=candidate, entry_sha256=candidate,
                parent_candidate_sha256=PARENT, parent_entry_sha256=ENTRY,
                input_binding_sha256=sha(OUT / 'inputs.json'),
                launcher_sha256=sha(SCRIPT), inherited_preflight=str(pre),
                preflight_sha256=sha(pre / 'status.json'),
                preflight_evidence_count=len(state['evidence_sha256']),
                input_identity_count=len(binding['files']))


def experiment_command(folder):
    return [PY, '-B', '-u', '-m', 'candidates.ig_sage_host_opt_v1.controller',
            '--mode', 'representative', '--model', 'sage',
            '--gpu', '2', '--output', str(folder / 'experiment')]


def validate_folder(folder):
    folder = folder.resolve()
    require(folder.parent == OUT / 'native', 'Unexpected output parent')
    return folder


def admission():
    require(host() >= 320 * 2**30, 'IG requires 320 GiB available host memory')
    gpu = subprocess.check_output([
        'nvidia-smi', '-i', '2',
        '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'],
        text=True, timeout=15).strip().split(',')
    require(len(gpu) == 2 and gpu[0].strip() == GPU_UUID and
            int(gpu[1].strip()) < 1024, 'GPU 2 identity/idle check failed')
    apps = subprocess.check_output([
        'nvidia-smi', '-i', '2', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    require(not apps.strip(), 'GPU 2 has another compute process')
    return dict(gpu_uuid=GPU_UUID, gpu_used_mib=int(gpu[1]),
                host_available_bytes=host(), checked_unix=time.time())


def stop(signum, frame):
    raise KeyboardInterrupt('Service interrupted')


def worker(folder):
    folder = validate_folder(folder)
    require(os.geteuid() == 0 and os.environ.get('TMUX'),
            'Worker requires local sudo and a real tmux session')
    child = None
    code = 1
    receipt = dict(passed=False, complete=False, started_unix=time.time())
    try:
        require(identities() == read(folder / 'identity.json'), 'Launch identity changed')
        command = experiment_command(folder)
        write(folder / 'experiment_command.json', command)
        with (folder / 'native.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment(2),
                                     stdout=log, stderr=subprocess.STDOUT)
            receipt['controller_pid'] = child.pid
            write(folder / 'worker_status.json', dict(receipt, stage='native_controller'))
            code = child.wait()
        receipt['controller_returncode'] = code
        require(code == 0, 'IG controller failed; inspect native.log and experiment/status.json')
        from candidates.ig_sage_host_opt_v1.summarize import load_formal
        value = load_formal(folder / 'experiment')
        require(value == read(folder / 'experiment/profile_summary/summary.json'),
                'Summary rebuild differs')
        require(identities() == read(folder / 'identity.json'), 'Completion identity changed')
        state = read(folder / 'experiment/status.json')
        counts = {w['mode'] + '_' + w['arm']: len(read(
            folder / 'experiment' / (w['mode'] + '_' + w['arm'] + '_receipt.json'))['files'])
            for w in state['workers']}
        write(folder / 'completion_review.json', dict(
            passed=True, complete=True, evidence_files_by_worker=counts,
            workers=len(state['workers']), summary_rebuilt_equal=True,
            strict_resource_acceptance=value['strict_resource_acceptance'],
            variants=value['variants'],comparisons=value['comparisons'], source_unchanged=True,
            accuracy_claim=False, epoch_time_claim=False, finished_unix=time.time()))
        receipt.update(passed=True, complete=True)
        code = 0
    except BaseException as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        receipt.update(returncode=code, finished_unix=time.time())
        write(folder / 'worker_exit.json', receipt)
    return code


def supervise(folder):
    folder = validate_folder(folder)
    require(os.geteuid() == 0, 'Run with local sudo')
    require(not folder.exists(), 'Preserve previous output')
    folder.mkdir(parents=True)
    state = dict(passed=False, complete=False, started_unix=time.time(),
                 directory=str(folder), raw_ssd_writes=False)
    socket_dir = None
    tmux = None
    def save(stage, **extra):
        state.update(stage=stage, updated_unix=time.time(), **extra)
        write(folder / 'status.json', state)
        write(OUT / 'status.json', state)
    try:
        save('checking_source_and_resources')
        write(folder / 'identity.json', identities())
        # Prevent concurrent author PA 512 B runs. The original IG controller
        # and native worker retain their own shared controller/device locks.
        with contextlib.ExitStack() as stack:
            lock = stack.enter_context(open('/tmp/digit-pa-sage-512b-pair-controller.lock', 'a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(folder / 'admission.json', admission())
            socket_dir = Path(tempfile.mkdtemp(prefix='digit-ig-sage-host-opt-'))
            tmux = ['/usr/bin/tmux', '-S', str(socket_dir / 'tmux.sock'), '-f', '/dev/null']
            env = environment(2)
            env.pop('TMUX', None)
            subprocess.run(tmux + ['new-session', '-d', '-s', 'ig-sage', '-c', str(ROOT),
                PY, '-B', str(SCRIPT), '--worker', str(folder)], check=True, cwd=ROOT, env=env)
            save('native_controller', tmux_socket=str(socket_dir / 'tmux.sock'))
            deadline = time.monotonic() + 6 * 3600
            while True:
                alive = subprocess.run(tmux + ['has-session', '-t', 'ig-sage'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0
                if not alive:
                    require((folder / 'worker_exit.json').exists(), 'Worker ended without exit receipt')
                    receipt = read(folder / 'worker_exit.json')
                    require(receipt['returncode'] == 0 and receipt['passed'] and receipt['complete'],
                            'IG run failed; inspect worker_exit.json and native.log')
                    save('complete', passed=True, complete=True, finished_unix=time.time())
                    return 0
                require(time.monotonic() < deadline, 'Run exceeded six hours')
                time.sleep(1)
    except BaseException as error:
        save('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
             error=type(error).__name__ + ': ' + str(error), finished_unix=time.time())
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if tmux is not None:
            subprocess.run(tmux + ['kill-server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if socket_dir is not None:
            for path in socket_dir.iterdir():
                path.unlink()
            socket_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--supervise', type=Path)
    group.add_argument('--worker', type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.worker:
        return worker(args.worker)
    if args.supervise:
        return supervise(args.supervise)
    identity = identities()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + str(os.getpid())
    folder = OUT / 'native' / stamp
    unit = 'digit-ig-sage-host-opt-20260927-v1-' + stamp
    command = ['/usr/bin/systemd-run', '--unit=' + unit,
        '--property=Type=exec', '--property=WorkingDirectory=' + str(ROOT),
        '--property=UMask=0022', '--property=Restart=no', '--property=KillMode=control-group',
        '--property=TimeoutStopSec=90', '--property=LimitMEMLOCK=infinity',
        '--setenv=CUDA_VISIBLE_DEVICES=2', '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=LD_LIBRARY_PATH=' + str(ROOT / 'bam/build/lib'),
        PY, '-B', '-u', str(SCRIPT), '--supervise', str(folder)]
    plan = dict(command=command, unit=unit + '.service', output=str(folder),
        execute=args.execute, identity=identity, model='sage', arms=['digit_full'],variants=['legacy','affinity','combined','compact'],
        warmup_batches=20, measured_batches=300, measurement_windows=3,
        combined_smoke_batches=2, formal_workers=4, profile_modes=['host'], raw_ssd_writes=False, model_or_backend_changed=False,
        continuous_monitor='unchanged IG v5 persistent NVML',
        gpu=2, gpu_uuid=GPU_UUID, cpu_binding='per variant: unchanged or worker threads on logical CPU2',
        host_admission_gib=320, ae_changed=False, pa_512b_optimizations=False,
        separate_sampling_probe=dict(warmup=20,measured=8,feature_reads=0,model_updates=0))
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        require(os.geteuid() == 0, 'Run this launcher using local sudo')
        subprocess.run(command, check=True)
        write(OUT / 'launch.json', dict(plan, started_unix=time.time()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
