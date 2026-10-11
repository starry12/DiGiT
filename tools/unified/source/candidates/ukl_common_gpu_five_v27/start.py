"""Bounded five-pair UKL performance without stage instrumentation."""
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
from candidates.ukl_common_gpu_five_v27 import protocol as P
from candidates.ukl_common_gpu_five_v27.guard import collect, Guard, AdmissionWindow, KernelCursor
from candidates.ukl_common_gpu_five_v27.io_preflight import run_probe
from candidates.ukl_common_gpu_five_v27.runtime import RuntimeMonitor
from candidates.ukl_memory_ladder_v5 import start as legacy
from . import gpu as GPU
from .phase import Manager
from .lean_monitor import GPUObservation
from .terminal_state import ServiceEvidence


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(str(tmp), str(path))


def append(log, value):
    log.write(json.dumps(value)+'\n')
    log.flush()


def command(unit, out, selected):
    b = P.budget()
    device, _ = P.source_device()
    properties = ['Type=exec', 'User=root', 'WorkingDirectory='+str(ROOT),
        'RemainAfterExit=yes', 'Delegate=yes', 'MemoryAccounting=yes', 'IOAccounting=yes',
        'MemoryMax='+str(b['memory_max']), 'MemoryHigh='+str(b['memory_high']),
        'MemorySwapMax=0', 'LimitMEMLOCK='+str(b['memlock']), 'TasksMax=64',
        'CPUQuota=100%', 'RuntimeMaxSec='+str(b['runtime_seconds']),
        'TimeoutStopSec=90', 'KillMode=control-group', 'Restart=no',
        'IOWeight=10', 'IOSchedulingClass=best-effort', 'IOSchedulingPriority=7',
        'IOReadBandwidthMax='+device+' '+str(P.READ_RATE),
        'IOWriteBandwidthMax='+device+' '+str(P.WRITE_RATE),
        'Nice=19', 'NoNewPrivileges=yes', 'ProtectSystem=strict',
        'ProtectHome=read-only', 'ReadWritePaths='+str(out),
        'ReadOnlyPaths=/mnt/n0', 'DevicePolicy=closed', 'DeviceAllow=/dev/libnvm0 rw',
        'DeviceAllow=/dev/nvidia'+str(selected[0])+' rw', 'DeviceAllow=/dev/nvidiactl rw', 'DeviceAllow=/dev/nvidia-uvm rw',
        'InaccessiblePaths=-/mnt/n3 -/mnt/n4 -/mnt/n5',
        'UMask=0022']
    if P.STAGE=='digit':properties.append('CPUAffinity=2')
    args = ['/usr/bin/systemd-run', '--unit='+unit]
    args += ['--property='+p for p in properties]
    env = dict(UKL_V27_VARIANT=P.VARIANT,UKL_V15_BOUNDED_WORKER='1',UKL_V27_BOUNDED_WORKER='1', UKL_V27_OUTPUT=str(out),
               CUDA_VISIBLE_DEVICES=selected[1], UKL_SELECTED_GPU_INDEX=str(selected[0]), CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(ROOT/'bam/build/lib'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DGLBACKEND='pytorch')
    args += ['--setenv='+k+'='+v for k, v in env.items()]
    return args+[P.PYTHON, '-B', '-u', '-m', 'candidates.ukl_common_gpu_five_v27.bootstrap',
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
    # SIGTERM is caught by the worker; chunk boundaries unwind and unregister.
    # Do not block the monitor while waiting. systemd retains a bounded 90 s stop.
    legacy.run(['/usr/bin/systemctl', 'stop', '--no-block', unit])


def _execute():
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
        unit = 'digit-ukl-common-gpu-v27-'+P.VARIANT+'-'+out.name.replace('_','-')+'.service'
        cg = '/system.slice/'+unit
        phases = Manager(cg,out)
        state, error, launched = {}, None, False
        service_evidence=ServiceEvidence()
        def show_state():return service_evidence.observe(legacy.show(unit))
        runtime = None
        gpu_monitor=GPU.Monitor(P.SELECTED,cg)
        gpu_task=None;gpu_latest=[None]
        io_preflight = {}
        post_ok, probes, kernel_ok = True, [], True
        original_term = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise RuntimeError('Controller termination requested')

        signal.signal(signal.SIGTERM, terminate)
        with (out/'monitor.jsonl').open('x') as log, (out/'kernel.jsonl').open('x') as kernel_log:
            gpu_identity=GPU.kernel_identity(P.SELECTED)
            kernel = KernelCursor(selected_uuid=gpu_identity['uuid'],selected_pci=gpu_identity['pci'])

            def sample(own):
                value = collect(cg if own else None)
                if own:phases.attach(value)
                from .numa_monitor import snapshot
                value['numa_diagnostics']=snapshot(cg if own else None)
                return value

            def kernel_poll(initial=False):
                nonlocal kernel_ok
                evidence = kernel.establish() if initial else kernel.poll()
                append(kernel_log, dict(time=time.time(), **evidence))
                if not evidence['ok']:
                    kernel_ok = False
                    raise RuntimeError('Kernel monitor/alert: '+str(evidence.get('error')))

            def gpu_poll(force=False):
                nonlocal gpu_task
                if gpu_task is None:
                    gpu_task=GPUObservation(gpu_monitor,interval=10).start()
                for row in gpu_task.check():append(log,dict(phase='gpu_observation',**row))
                gpu_latest[0]=gpu_task.inspect()


            try:
                kernel_poll(initial=True)
                io_preflight = run_probe(out, unit)
                kernel_poll()
                _, device = P.source_device()
                admission = AdmissionWindow(b['host_min'], source_devices=(device,))
                deadline = time.monotonic()+b['max_quiet_wait_seconds']
                print('Waiting for 30 continuous quiet seconds (at most 5 minutes): '+str(out), flush=True)
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
                gpu_index, uuid = P.SELECTED
                GPU.check_idle(P.SELECTED)
                write(out/'before.json', dict(source=source, budget=b, gpu=gpu_index,
                    uuid=uuid, manifest_sha256=digest, last_admission=observation,
                    monitor_policy=dict(gpu_interval_seconds=10,query_timeout_fatal=False,attribution="selected_gpu_uuid_pci",unattributed_thread_timeout="record_only",admission_checks=True), stage=P.STAGE,
                    predecessor=predecessor,pair_id=P.PAIR_ID,selected_gpu=list(P.SELECTED)))
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
                    raise RuntimeError('Admission changed before launch')
                GPU.check_idle(P.SELECTED)
                P.binding()
                if P.require_predecessor() != predecessor:
                    raise RuntimeError('Predecessor evidence changed before launch')
                write(out/'controller_lease.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
                args = command(unit, out, P.SELECTED)
                write(out/'command.json', dict(command=args))
                launched = True
                legacy.run(args)
                gpu_poll()
                print('Bounded %s stage launched on GPU %s' % (P.STAGE,gpu_index), flush=True)
                guard = Guard(phases=phases)
                runtime = RuntimeMonitor(guard, cg, worker_passed)
                deadline = time.monotonic()+b['runtime_seconds']+20
                while time.monotonic() < deadline:
                    state = show_state()
                    phases.poll(state)
                    observation = sample(True)
                    outcome = runtime.observe(state, observation,
                        show_state, lambda: read_worker_report(out))
                    decision = outcome['decision']
                    append(log, dict(phase='runtime', sample=observation,
                        state_before=state, **outcome))
                    state = outcome['state_after']
                    # Host alarms stop immediately; auxiliary reads never delay STOP.
                    if not decision['ok']:raise RuntimeError('Runtime guard: '+str(decision['reasons']))
                    kernel_poll()
                    gpu_poll()
                    if outcome['terminal']:
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('Worker deadline exceeded')
            except BaseException as failure:
                error = repr(failure)
                if launched:
                    try:
                        service_evidence.stopping(show_state())
                        write(out/'state_before_stop.json',service_evidence.receipt())
                        request_stop(unit, out)
                    except BaseException as stop_error:
                        error += '; stop='+repr(stop_error)
            finally:
                signal.signal(signal.SIGTERM, original_term)
                if launched:
                    # No unconditional wait on potentially uninterruptible tasks.
                    stop_deadline = time.monotonic()+2
                    while time.monotonic() < stop_deadline:
                        try:
                            state = show_state()
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
                        gpu_poll()
                    except Exception as post_error:
                        post_ok = False
                        append(log, dict(phase='post_error', time=time.time(), error=repr(post_error)))
                    time.sleep(1)
                if launched:
                    try:state=show_state()
                    except Exception as e:post_ok=False;error=str(error)+'; final_state='+repr(e)
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
                final_lookup=dict(state)
                state=service_evidence.worker_state()
                passed = (launched and not error and worker_passed(state, report)
                          and kernel_ok and post_ok and len(afterprobes) == 2
                          and all(p['passed'] for p in afterprobes))
                if gpu_task is not None:gpu_task.stop()
                try:
                    GPU.wait_released(P.SELECTED,record=lambda v:append(log,dict(phase='gpu_release',observation=v)))
                    gpu_released=True
                except Exception as gpu_error:
                    gpu_released=False;passed=False;error=str(error)+'; gpu_release='+repr(gpu_error)
                result = dict(passed=bool(passed), worker_launched=launched, error=error,
                    stage=P.STAGE, predecessor=predecessor, graph_sampling_enabled=not P.small(), storage_write_enabled=False, model_enabled=not P.small(), features_enabled=True,
                    pair_id=P.PAIR_ID,selected_gpu=list(P.SELECTED),gpu_peak_used_mib=gpu_monitor.peak,gpu_released=gpu_released,gpu_monitor_policy=dict(interval_seconds=10,query_errors_fatal=False,conflicts_fatal=True),
                    unit=unit, worker_state=state, final_lookup=final_lookup, service_state_evidence=service_evidence.receipt(),
                    worker_failure=report if report.get('scope')=='worker_failure' else None,
                    worker=report, kernel_monitor_ok=kernel_ok,
                    post_guard_passed=post_ok, post_seconds=time.time()-post_start,
                    post_probes=afterprobes, manifest_sha256=digest,
                    io_preflight_passed=io_preflight.get('passed') is True,
                    runtime_valid_samples=runtime.valid_runtime_samples if runtime else 0,
                    full_graph_load=not P.small(),full_graph_enabled=not P.small(), raw_ssd_access=(report.get('io',{}).get('submitted_commands',0)>0 or bool(report.get('cache',{}).get('preload_io',{}).get('submitted_commands',0))) if report and report.get('scope')!='worker_failure' else None,raw_ssd_writes=False,timed_performance_enabled=True)
                write(out/'acceptance.json', result)
                print(json.dumps(dict(output=str(out), passed=result['passed'], error=error,
                                      worker_error=report.get('error') if report.get('scope')=='worker_failure' else None)), flush=True)
                if state.get('ActiveState') == 'active' and state.get('SubState') == 'exited':
                    legacy.run(['/usr/bin/systemctl', 'stop', unit])
        if not result['passed']:
            raise RuntimeError('Arm not accepted: '+str(out))
        return dict(path=str(out/'acceptance.json'),sha256=hashlib.sha256((out/'acceptance.json').read_bytes()).hexdigest(),arm=P.STAGE,seconds=report['seconds'],pair_id=P.PAIR_ID)


def execute():
    # Actual preprocessing shares the accepted loader lock. Qualification does
    # not take this lock, so it cannot interrupt the current UKL generation.
    from contextlib import ExitStack
    from ae.common import device_idle
    with ExitStack() as stack:
        for path in (P.UKL.LOCK,Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),Path('/tmp/digit-pa-sage-libnvm0.lock'),ROOT/'ssd_state/libnvm0.prepare.lock',Path('/run/digit-ae-selfservice/exclusive.lock')):
            handle=stack.enter_context(path.open('r+' if path.exists() else 'x+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device_idle()
        P.SELECTED=GPU.select(GPU.snapshot())
        run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        from .series import run_series
        def run_one(number,stage):
            P.PAIR_ID=run_id+'_r'+str(number)
            P.configure_stage(stage)
            device_idle();GPU.check_idle(P.SELECTED)
            return _execute()
        result=run_series(P.OUT,run_id,list(P.SELECTED),run_one,rounds=P.ROUNDS)
        from .comparison import review
        write(P.OUT/('comparison_'+run_id+'.json'),review(result))
        return result


def main():
    from .cli import main as cli
    cli()

if __name__ == '__main__':main()
