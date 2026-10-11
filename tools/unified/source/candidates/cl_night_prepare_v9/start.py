"""Bounded CPU-only UKL Freq/BFS and synthetic feature-file preparation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from candidates.cl_night_prepare_v9 import protocol as P
from candidates.cl_night_prepare_v9.guard import collect, Guard, AdmissionWindow, KernelCursor
from candidates.cl_night_prepare_v9.io_preflight import run_probe
from candidates.cl_night_prepare_v9.runtime import RuntimeMonitor
from candidates.ukl_memory_ladder_v5 import start as legacy


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(str(tmp), str(path))


def append(log, value):
    log.write(json.dumps(value)+'\n')
    log.flush()


def command(unit, out, uuid):
    b = P.budget()
    device, _ = P.source_device()
    properties = ['Type=exec', 'User=embed', 'WorkingDirectory='+str(ROOT),
        'RemainAfterExit=yes', 'MemoryAccounting=yes', 'IOAccounting=yes',
        'MemoryMax='+str(b['memory_max']), 'MemoryHigh='+str(b['memory_high']),
        'MemorySwapMax=0', 'LimitMEMLOCK='+str(b['memlock']), 'TasksMax=64',
        'CPUQuota=100%', 'RuntimeMaxSec='+str(b['runtime_seconds']),
        'TimeoutStopSec=90', 'KillMode=control-group', 'Restart=no',
        'IOWeight=10', 'IOSchedulingClass=best-effort', 'IOSchedulingPriority=7',
        'IOReadBandwidthMax='+device+' '+str(P.READ_RATE),
        'IOWriteBandwidthMax='+device+' '+str(P.WRITE_RATE),
        'Nice=19', 'NoNewPrivileges=yes', 'ProtectSystem=strict',
        'ProtectHome=read-only', 'ReadWritePaths='+str(out)+' '+str(P.DATA_ROOT/out.name),
        'ReadOnlyPaths=/mnt/n0', 'DevicePolicy=closed', 'InaccessiblePaths=-/mnt/n3 -/mnt/n4 -/mnt/n5 -/dev/libnvm0 -/dev/nvidiactl -/dev/nvidia-uvm -/dev/nvidia0 -/dev/nvidia1 -/dev/nvidia2 -/dev/nvidia3',
        'UMask=0022']
    args = ['/usr/bin/systemd-run', '--unit='+unit]
    args += ['--property='+p for p in properties]
    env = dict(CL_NIGHT_V9_WORKER='1', CL_NIGHT_V9_OUTPUT=str(out),
               CL_NIGHT_V9_DATA=str(P.DATA_ROOT/out.name), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DGLBACKEND='pytorch')
    args += ['--setenv='+k+'='+v for k, v in env.items()]
    return args+[P.PYTHON, '-B', '-u', '-m', 'candidates.cl_night_prepare_v9.worker',
                 '--stage', P.STAGE]


def worker_passed(state, report):
    return P.worker_passed(state, report)


def read_worker_report(out):
    report = json.loads((out/'worker.json').read_text())
    if not isinstance(report, dict):
        raise ValueError('Worker report must be an object')
    return report


def request_stop(unit, out):
    write(out/'STOP', dict(time=time.time(), reason='controller_guard'))
    # First request cooperative cleanup. systemd is scoped to this worker only.
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if legacy.exited(legacy.show(unit)):
            return
        time.sleep(0.25)
    legacy.run(['/usr/bin/systemctl', 'stop', '--no-block', unit])


def execute():
    if os.geteuid() != 0:
        raise RuntimeError('sudo required for systemd limits and complete kernel/IO evidence')
    digest = P.verify_manifest()
    source = P.binding()
    b = P.budget()
    with P.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        predecessor = P.require_predecessor()
        out = P.OUT / ('stage_'+P.STAGE+'_run_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
        out.mkdir()
        account = pwd.getpwnam('embed')
        os.chown(str(out), account.pw_uid, account.pw_gid)
        unit = 'digit-cl-night-v9-'+out.name.replace('_','-')+'.service'
        cg = '/system.slice/'+unit
        state, error, launched = {}, None, False
        runtime = None
        io_preflight = {}
        post_ok, probes, kernel_ok = True, [], True
        original_term = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise RuntimeError('Controller termination requested')

        signal.signal(signal.SIGTERM, terminate)
        with (out/'monitor.jsonl').open('x') as log, (out/'kernel.jsonl').open('x') as kernel_log:
            kernel = KernelCursor()

            def sample(own):
                value = collect(cg if own else None)
                return value

            def kernel_poll(initial=False):
                nonlocal kernel_ok
                evidence = kernel.establish() if initial else kernel.poll()
                append(kernel_log, dict(time=time.time(), **evidence))
                if not evidence['ok']:
                    kernel_ok = False
                    raise RuntimeError('Kernel monitor/alert: '+str(evidence.get('error')))

            try:
                kernel_poll(initial=True)
                io_preflight = run_probe(out, unit)
                kernel_poll()
                _, device = P.source_device()
                admission = AdmissionWindow(b['host_min'], source_devices=(device,))
                deadline = time.monotonic()+b['max_quiet_wait_seconds']
                print('Waiting for 30 quiet seconds (at most 30 minutes): '+str(out), flush=True)
                while time.monotonic() < deadline:
                    observation = sample(False)
                    decision = admission.check(observation)
                    append(log, dict(phase='admission', sample=observation, decision=decision))
                    kernel_poll()
                    if not decision['ok']:
                        raise RuntimeError('Admission monitor failed: '+str(decision['reasons']))
                    if decision['ready']:
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('No quiet admission window; no worker launched')
                gpu_index, uuid = None, ''
                write(out/'before.json', dict(source=source, budget=b, gpu=gpu_index,
                    uuid=uuid, manifest_sha256=digest, last_admission=observation,
                    controller_lease=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]), stage=P.STAGE,
                    predecessor=predecessor))
                probes = [legacy.probe(ROOT/'results', out, 'root_before'),
                          legacy.probe(Path('/mnt/n0'), out, 'n0_before')]
                write(out/'probes_before.json', probes)
                if not all(p['passed'] for p in probes):
                    raise RuntimeError('Pre-run filesystem probe failed')
                observation = sample(False)
                decision = admission.check(observation)
                append(log, dict(phase='admission_final', sample=observation, decision=decision))
                kernel_poll()
                if not decision['ready']:
                    raise RuntimeError('Admission/GPU changed before launch')
                P.binding()
                if P.require_predecessor() != predecessor:
                    raise RuntimeError('Predecessor evidence changed before launch')
                P.DATA_ROOT.mkdir(exist_ok=True)
                if P.DATA_ROOT.is_symlink() or P.DATA_ROOT.resolve()!=P.DATA_ROOT:
                    raise RuntimeError('Data root symlink forbidden')
                if P.DATA_ROOT.stat().st_dev != source['indices']['identity'][0]:
                    raise RuntimeError('Output filesystem differs from graph source')
                if shutil.disk_usage(P.DATA_ROOT).free < sum(P.expected_files().values())+64*P.GIB:
                    raise RuntimeError('Insufficient output space plus 64 GiB reserve')
                target=P.DATA_ROOT/out.name
                target.mkdir();os.chown(str(target),account.pw_uid,account.pw_gid)
                args = command(unit, out, uuid)
                write(out/'command.json', dict(command=args))
                launched = True
                legacy.run(args)
                print('Bounded CPU preparation stage %s launched; no GPU/raw SSD access' % P.STAGE, flush=True)
                guard = Guard()
                runtime = RuntimeMonitor(guard, cg, worker_passed)
                deadline = time.monotonic()+b['runtime_seconds']+20
                while time.monotonic() < deadline:
                    state = legacy.show(unit)
                    observation = sample(True)
                    outcome = runtime.observe(state, observation,
                        lambda: legacy.show(unit), lambda: read_worker_report(out))
                    decision = outcome['decision']
                    append(log, dict(phase='runtime', sample=observation,
                        state_before=state, **outcome))
                    state = outcome['state_after']
                    kernel_poll()
                    if not decision['ok']:
                        raise RuntimeError('Runtime guard: '+str(decision['reasons']))
                    if outcome['terminal']:
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('Worker deadline exceeded')
            except BaseException as failure:
                error = repr(failure)
                if launched:
                    try:
                        request_stop(unit, out)
                    except BaseException as stop_error:
                        error += '; stop='+repr(stop_error)
            finally:
                signal.signal(signal.SIGTERM, original_term)
                if launched:
                    # No unconditional wait on potentially uninterruptible tasks.
                    stop_deadline = time.monotonic()+20
                    while time.monotonic() < stop_deadline:
                        try:
                            state = legacy.show(unit)
                            if legacy.exited(state):
                                break
                        except Exception as show_error:
                            error = str(error)+'; show='+repr(show_error)
                            break
                        time.sleep(1)
                post_guard = Guard(require_cgroup=False)
                deadline = time.monotonic()+(b['failure_post_seconds'] if error else b['post_seconds'])
                post_start = time.time()
                while time.monotonic() < deadline:
                    try:
                        observation = sample(False)
                        decision = post_guard.check(observation)
                        append(log, dict(phase='post', sample=observation, decision=decision))
                        post_ok = post_ok and decision['ok']
                        kernel_poll()
                    except Exception as post_error:
                        post_ok = False
                        append(log, dict(phase='post_error', time=time.time(), error=repr(post_error)))
                    time.sleep(1)
                if launched:
                    try:state=legacy.show(unit)
                    except Exception as exc:error=str(error)+'; final_state='+repr(exc);post_ok=False
                afterprobes = []
                if launched and legacy.exited(state):
                    try:
                        afterprobes = [legacy.probe(ROOT/'results', out, 'root_after'),
                                       legacy.probe(Path('/mnt/n0'), out, 'n0_after')]
                    except Exception as probe_error:
                        post_ok = False
                        append(log, dict(phase='probe_error', error=repr(probe_error)))
                try:
                    observation = sample(False)
                    decision = post_guard.check(observation)
                    append(log, dict(phase='final', sample=observation, decision=decision))
                    post_ok = post_ok and decision['ok']
                    kernel_poll()
                except Exception as final_error:
                    post_ok = False
                    append(log, dict(phase='final_error', error=repr(final_error)))
                try:
                    report = read_worker_report(out)
                except (OSError, ValueError) as report_error:
                    report = {}
                    if launched:
                        error = str(error)+'; worker_report='+repr(report_error)
                passed = (launched and not error and worker_passed(state, report)
                          and kernel_ok and post_ok and len(afterprobes) == 2
                          and all(p['passed'] for p in afterprobes))
                result = dict(passed=bool(passed), worker_launched=launched, error=error,
                    stage=P.STAGE, predecessor=predecessor, graph_sampling_enabled=P.STAGE=='graph', model_enabled=False, features_enabled=P.STAGE.startswith('features-'),
                    unit=unit, worker_state=state, worker=report, kernel_monitor_ok=kernel_ok,
                    post_guard_passed=post_ok, post_seconds=time.time()-post_start,
                    post_probes=afterprobes, manifest_sha256=digest,
                    io_preflight_passed=io_preflight.get('passed') is True,
                    runtime_valid_samples=runtime.valid_runtime_samples if runtime else 0,
                    full_graph_load=P.STAGE=='graph',
                    full_graph_enabled=P.STAGE=='graph', raw_ssd_access=False)
                write(out/'acceptance.json', result)
                print(json.dumps(dict(output=str(out), passed=result['passed'], error=error)), flush=True)
                if state.get('ActiveState') == 'active' and state.get('SubState') == 'exited':
                    legacy.run(['/usr/bin/systemctl', 'stop', unit])
        if not result['passed']:
            raise RuntimeError('Stage failed: '+str(out))
        return out,result


def main():
    from .night import main as cli
    cli()

if __name__=='__main__':main()
