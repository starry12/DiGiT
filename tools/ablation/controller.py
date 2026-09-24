"""Fresh four-arm AE run over an immutable namespace-mounted author snapshot."""
import fcntl, importlib, json, os, signal, subprocess, sys, time, uuid
from pathlib import Path
ROOT = Path('/home/embed/digit')
CONTROL = Path('/srv/digit-ae/admin/ablation_v1')
OUTPUTS = Path('/srv/digit-ae/ablation-results')
ARMS = ('gids', 'gr', 'ns', 'digit_full')
sys.path.insert(0, str(ROOT))
from ae.common import require, read, write, sha, check_device
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.monitor_validation import monitor_evidence
from candidates.pa_sage_ablation_graph_v4.validation import matrix_check

def identity(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)

def package(arm): return 'candidates.pa_sage_ablation_' + ('graph_v4' if arm == 'gr' else 'cache_v3')

def bind_inputs(output):
    snapshot = json.loads((CONTROL / 'snapshot_manifest.json').read_text())
    for name, digest in snapshot['files'].items():
        require(sha(ROOT / name) == digest, 'Snapshot changed: ' + name)
    template = json.loads((CONTROL / 'inputs_template.json').read_text())
    files = {}
    for name, desc in template['files'].items():
        path = ROOT / name
        actual = identity(path)
        if name in snapshot['files']:
            require(sha(path) == desc['sha256'], 'Copied metadata changed: ' + name)
        else:
            require(actual == desc['identity'], 'Prepared data changed: ' + name)
        files[name] = dict(sha256=desc['sha256'], identity=actual)
    bindings = {}
    for key in ('gids', 'gr'):
        common = importlib.import_module(package(key) + '.common')
        value = dict(template, files=files, candidate_sha256=common.verify(), protocol_sha256=sha(common.P))
        write(output / ('inputs_' + key + '.json'), value); bindings[key] = value
    return bindings

def execute(selftest=False):
    require(os.geteuid() == 0 and __debug__, 'Root service without Python optimization required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'GPU 2 only')
    # The alias must resolve to the sealed snapshot, never the mutable author checkout.
    require(sha(ROOT / 'snapshot_identity.json') == sha(CONTROL / 'snapshot_identity.json'), 'Snapshot namespace missing')
    output = OUTPUTS / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:12])
    output.mkdir(); state = dict(passed=False, complete=False, stage='admission', pid=os.getpid(),
        started_unix=time.time(), workers=[], selftest=selftest, gpu=2,
        snapshot_sha256=sha(CONTROL / 'snapshot_manifest.json'), raw_ssd_writes=False)
    def save(**kw): state.update(kw, updated_unix=time.time()); write(output / 'status.json', state)
    save(); write(CONTROL / ('state/selftest.json' if selftest else 'state/latest.json'), dict(output=str(output), request_origin='fixed AE service; systemd does not record caller identity here', authorized_account='atc27_ae', started_unix=state['started_unix']))
    locks = []; monitor = None; child = None
    def stop(*args): raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        for name in ('/run/digit-ae-selfservice/exclusive.lock', '/tmp/digit-pa-bidir-controller.lock'):
            handle = open(name, 'a+'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
        save(stage='verifying_snapshot_and_inputs'); bindings = bind_inputs(output)
        save(stage='verifying_isolated_runtime_imports')
        import_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        with (output / 'runtime_imports.log').open('x') as log:
            subprocess.run([sys.executable, '-I', '-B', str(CONTROL / 'check_imports.py'), str(ROOT)],
                cwd='/tmp', env=import_env, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
        if selftest:
            save(stage='complete', passed=True, complete=True, native_acceptance=False, finished_unix=time.time())
            return
        gpu = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        require(gpu == 'GPU-927ce617-743a-4bfe-6a60-8a8311cfc703' and gpu not in apps, 'GPU 2 occupied or changed')
        for mode in ('smoke', 'full'):
            reports = {}
            for arm in ARMS:
                save(stage=mode + '_' + arm); check_device()
                folder = output / mode / arm; folder.parent.mkdir(exist_ok=True)
                mon_dir = output / (mode + '_monitor_' + arm); mon_dir.mkdir()
                monitor = ExternalMonitor(mon_dir / 'external_gpu'); ready = monitor.start()
                key = 'gr' if arm == 'gr' else 'gids'
                command = [sys.executable, '-B', '-u', '-m', package(arm) + '.worker', '--arm', arm,
                    '--output', str(folder), '--binding', str(output / ('inputs_' + key + '.json'))]
                if mode == 'smoke': command.append('--smoke')
                w = dict(mode=mode, arm=arm, status='running', started_unix=time.time())
                state['workers'].append(w)
                with (output / (mode + '_' + arm + '.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    w['pid'] = child.pid; save(); released = False; deadline = None
                    while child.poll() is None:
                        receipt = folder / 'worker_ready.json'
                        if receipt.exists() and not released:
                            require(read(receipt)['passed'], 'Worker acceptance failed')
                            code = monitor.stop(); monitor = None
                            write(folder / 'release_worker.json', dict(monitor_stopped=True, monitor_returncode=code))
                            released = True; deadline = time.monotonic() + 180
                        if deadline: require(time.monotonic() < deadline, 'Worker teardown timeout')
                        time.sleep(.5)
                    rc = child.wait(); child = None
                w.update(returncode=rc, status='complete' if rc == 0 else 'failed', finished_unix=time.time()); save()
                require(rc == 0 and released and code == 0, 'Worker/monitor failed')
                require(sha(folder / 'report.json') == read(folder / 'worker_ready.json')['report_sha256'], 'Report changed')
                r = read(folder / 'report.json')
                require(read(mon_dir / 'external_gpu/summary.json')['errors'] == [], 'Monitor query errors')
                ms = dict(pid=state['pid'], workers=[w], external_monitor=ready, external_monitor_returncode=code)
                r['external_monitor'] = monitor_evidence(mon_dir, ms, arm, read(folder / 'resources.json'))
                validation = importlib.import_module(package(arm) + '.validation')
                validation.report_check(r, arm, mode == 'smoke', bindings[key], folder)
                require(r['input_binding_sha256'] == sha(output / ('inputs_' + key + '.json')), 'Input receipt changed')
                write(output / (mode + '_' + arm + '_accepted.json'), r); reports[arm] = r
            summary = matrix_check(reports, mode == 'smoke')
            summary.update(snapshot_sha256=state['snapshot_sha256'], reused_full_arms=[],
                report_sha256={arm: sha(output / (mode + '_' + arm + '_accepted.json')) for arm in ARMS},
                candidate_sha256={arm: r['candidate_sha256'] for arm, r in reports.items()})
            write(output / (mode + '_summary.json'), summary); save(stage=mode + '_accepted')
        for binding in bindings.values():
            for name, desc in binding['files'].items(): require(identity(ROOT / name) == desc['identity'], 'Input changed during run')
        check_device()
        save(stage='complete', passed=True, complete=True, summary_sha256=sha(output / 'full_summary.json'), finished_unix=time.time())
    except BaseException as exc:
        save(stage='failed', error=type(exc).__name__ + ': ' + str(exc)); raise
    finally:
        if monitor is not None: monitor.stop()
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=60)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for handle in locks: handle.close()

if __name__ == '__main__':
    require(sys.argv[1:] in ([], ['--selftest']), 'No arbitrary controller arguments')
    execute(selftest=sys.argv[1:] == ['--selftest'])
