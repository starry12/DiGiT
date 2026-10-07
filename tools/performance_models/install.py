#!/usr/bin/python3
"""Install eight model services and CPU selftests; never launch performance jobs."""
import argparse
import contextlib
import fcntl
import json
import os
import pwd
import time
from pathlib import Path
from install_support import atomic, invoke, pointer, read, record, require, sha, tree

H = Path(__file__).resolve().parent
ADMIN = Path('/srv/digit-ae/admin')
DEST = ADMIN / 'multimodel_v2'
POINTER = Path('/srv/digit-ae/reviewer-current')
SUDO = Path('/etc/sudoers.d/digit-ae-multimodel')
LOCKS = ['/run/digit-ae-selfservice/exclusive.lock', '/tmp/digit-pa-sage-experiment.lock',
         '/tmp/digit-pa-bidir-controller.lock', '/tmp/digit-pa-sage-libnvm0.lock',
         '/home/embed/digit/ssd_state/libnvm0.prepare.lock',
         '/home/embed/digit/results/ukl_runtime_prepare_20261001_v6/smoke.lock']


def verify():
    for name, digest in read(H / 'installation_manifest.json')['files'].items():
        p = H / name
        require(not p.is_symlink() and sha(p) == digest, 'Prepared package changed: ' + name)
    require(read(H / 'validation.json')['passed'], 'Preparation validation did not pass')
    return read(H / 'install_plan.json')


@contextlib.contextmanager
def shared_locks():
    with contextlib.ExitStack() as stack:
        for name in LOCKS:
            p = Path(name)
            require(p.is_file() and not p.is_symlink(), 'Missing trusted lock: ' + name)
            f = stack.enter_context(p.open('rb'))
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_server_state(plan, configs, installed=False):
    units = {name: H / 'payload' / c['key'] / 'units' / name
             for c in configs for name in (c['unit'], c['selftest'])}
    for p, before in plan['before'].items():
        prepared = units.get(Path(p).name)
        # atomic() installs units as 0644 regardless of the author's umask.
        # Compare the destination against that contract, not staging metadata.
        expected = dict(sha256=sha(prepared), mode=0o644) if prepared is not None else before
        actual = record(p)
        # Retry a partial installation only when the already-created unit is
        # byte-for-byte identical. All reviewer activation state stays guarded.
        allowed = [expected] if installed else [before, expected]
        require(actual in allowed, 'Server target changed: ' + p
                + '; expected=' + repr(allowed) + '; actual=' + repr(actual))


def execute():
    plan = verify()
    require(os.geteuid() == 0, 'Use sudo for the fixed server installation')
    require(pwd.getpwnam('atc27_ae').pw_uid == plan['reviewer_uid'], 'Reviewer account changed')
    os.umask(0o022)
    for directory in (ADMIN.parent, ADMIN, Path('/srv/digit-ae/reviewer_releases')):
        st = directory.stat()
        require(st.st_uid == 0 and not st.st_mode & 0o022, 'Untrusted server parent')
    configs = read(H / 'payload/config.json')['services']
    target = Path('/srv/digit-ae/reviewer_releases') / ('source_' + plan['reviewer_sha256'][:16])
    # A separate installation lock remains held during namespace selftests.
    # IG's CPU selftest acquires the shared experiment lock itself.
    with (ADMIN / '.multimodel-install.lock').open('a+') as install_lock:
        fcntl.flock(install_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with shared_locks():
            verify_server_state(plan, configs)
            for p, digest in read(H / 'parent_controls.json').items():
                require(sha(p) == digest, 'Parent control changed: ' + p)
            for c in configs:
                for unit in (c['unit'], c['selftest']):
                    status = invoke(['/usr/bin/systemctl', 'show', unit, '--property=ActiveState', '--value'], check=False)
                    require(status.stdout.strip() not in ('active', 'activating', 'deactivating'), 'Model service busy')
            invoke(['/usr/sbin/visudo', '-cf', H / 'sudoers'])
            tree(H / 'payload', DEST)
            tree(H / 'reviewer', target)
            for c in configs:
                for p in (Path(c['output']), Path(c['output']) / 'probes', Path(c['control']) / 'state'):
                    require(not p.is_symlink(), 'Symlink state directory')
                    p.mkdir(parents=True, exist_ok=True)
                    p.chmod(0o755)
                    require(p.stat().st_uid == 0, 'Untrusted state directory')
                for p in (Path(c['control']) / 'units').glob('*.service'):
                    dst = Path('/etc/systemd/system') / p.name
                    require(not dst.exists() or sha(dst) == sha(p), 'Existing unit differs')
                    atomic(dst, p.read_bytes())
            invoke(['/usr/bin/systemctl', 'daemon-reload'])
        checks = {}
        for c in configs:
            began = time.time()
            print('CPU namespace check: ' + c['key'], flush=True)
            result = invoke(['/usr/bin/systemctl', 'start', '--wait', c['selftest']], check=False)
            state = invoke(['/usr/bin/systemctl', 'show', c['selftest'], '--property=ActiveState,Result,ExecMainStatus,MainPID'])
            parsed = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
            receipt = Path(c['control']) / 'state/selftest.json'
            if result.returncode or parsed.get('Result') != 'success' or parsed.get('ExecMainStatus') != '0':
                log = invoke(['/usr/bin/journalctl', '-u', c['selftest'], '-n', '60', '--no-pager'], check=False)
                raise RuntimeError('New CPU selftest failed; reviewer pointer unchanged.\n' + log.stdout)
            require(receipt.exists() and receipt.stat().st_mtime >= began and read(receipt)['passed'], 'Fresh selftest receipt required')
            checks[c['key']] = dict(service=parsed, receipt=read(receipt), receipt_sha256=sha(receipt))
        with shared_locks():
            # Abort activation if another deployment changed during the checks.
            verify_server_state(plan, configs, installed=True)
            invoke(['/usr/bin/python3', '-I', '-B', target / 'tools/reviewer/test_cli.py'])
            backup = ADMIN / 'reviewer_backups' / ('multimodel_' + time.strftime('%Y%m%d_%H%M%S'))
            backup.mkdir()
            changed = [SUDO, ADMIN / 'reviewer_package.json', DEST / 'installation.json']
            old = {p: p.read_bytes() if p.exists() else None for p in changed}
            old_pointer = os.readlink(POINTER)
            for p, data in old.items():
                if data is not None:
                    atomic(backup / p.name, data)
            atomic(backup / 'before.json', (json.dumps(dict(pointer=old_pointer, absent=[str(p) for p, b in old.items() if b is None]), indent=2) + '\n').encode())
            try:
                atomic(SUDO, (H / 'sudoers').read_bytes(), 0o440)
                invoke(['/usr/sbin/visudo', '-c'])
                for c in configs:
                    for action in ('start', 'stop'):
                        invoke(['/usr/sbin/runuser', '-u', 'atc27_ae', '--', '/usr/bin/sudo', '-n', '-l', '/usr/bin/systemctl', '--no-block', action, c['unit']])
                denied = invoke(['/usr/sbin/runuser', '-u', 'atc27_ae', '--', '/usr/bin/sudo', '-n', '-l', '/bin/bash'], check=False)
                require(denied.returncode != 0, 'Overbroad reviewer sudo permission')
                pointer(target)
                receipt = dict(passed=True, installed=True, source=str(target), package_sha256=plan['reviewer_sha256'],
                               model_cpu_selftests=checks, native_acceptance=False, training_started=False, backup=str(backup))
                payload = (json.dumps(receipt, indent=2) + '\n').encode()
                atomic(DEST / 'installation.json', payload)
                atomic(ADMIN / 'reviewer_package.json', payload)
            except BaseException:
                pointer(old_pointer)
                for p, data in old.items():
                    if data is None:
                        p.unlink(missing_ok=True)
                    else:
                        atomic(p, data, 0o440 if p == SUDO else 0o644)
                raise
    print('Eight model services installed; fresh CPU namespace checks passed. No performance jobs started.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print(json.dumps(dict(ready=True, plan=verify(),training_started=False), indent=2))
