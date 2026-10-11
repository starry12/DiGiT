import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

HERE = Path(__file__).resolve().parent
CLI = Path('/usr/local/bin/digit-ae')
SUDOERS = Path('/etc/sudoers.d/digit-ae-unified')
GATE_SHA = 'a572edc41dd153abe03bdb649bd82d3aaca1c714f1a345733af6bfbbdec5dd83'
FILES = ('common.py', 'cli.py', 'runner.py', 'gate.py', 'install.py', 'enable.py', 'tests.py', 'USAGE.txt')

def service_text(dataset, mode):
    cap = 'MemoryMax=384G\n' if dataset in ('PA', 'UKS') else ''
    return ('[Unit]\nDescription=DiGiT unified ' + dataset + ' GraphSAGE ' + mode + '\n'
        'ConditionPathExists=' + str(ADMIN/'enabled'/dataset) + '\n'
        '[Service]\nType=exec\nUser=root\nGroup=root\nWorkingDirectory=' + str(ROOT) + '\n'
        'ExecStart=' + PYTHON + ' -I -B -u ' + str(ADMIN/'runner.py') + ' ' + dataset + ' ' + mode + '\n'
        'Environment=PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2\n'
        'Environment=LD_LIBRARY_PATH=' + str(ROOT/'bam/build/lib') + '\n'
        'UnsetEnvironment=PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD\n'
        'UMask=0022\nKillMode=mixed\nTimeoutStopSec=240\nTimeoutStartSec=infinity\n'
        'LimitMEMLOCK=infinity\nMemorySwapMax=0\nRestart=no\nNoNewPrivileges=yes\n'
        + cap + 'StandardOutput=journal\nStandardError=journal\n').encode()

def sudoers_text():
    return ''.join('atc27_ae ALL=(root) NOPASSWD: /usr/bin/systemctl --no-block ' + action + ' ' + unit(d, m) + '\n'
        for d in DATASETS for m in MODES for action in ('start', 'stop')).encode()

def payload():
    manifest = read(HERE/'payload.json')
    require(set(manifest['files']) == set(FILES), 'Wrong package file set')
    for name, digest in manifest['files'].items():
        require(sha(HERE/name) == digest, 'Package changed: ' + name)
    return manifest

def targets():
    values = {str(CLI): dict(data=(HERE/'gate.py').read_bytes(), mode=0o755),
              str(SUDOERS): dict(data=sudoers_text(), mode=0o440)}
    for d in DATASETS:
        for m in MODES:
            values[str(Path('/etc/systemd/system')/unit(d, m))] = dict(data=service_text(d, m), mode=0o644)
    return values

def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=60)

def idle():
    for d in DATASETS:
        for m in MODES:
            require(not active(service(d, m)), 'Unified reviewer service active; no changes made: ' + unit(d, m))

def check_sudoers(data):
    import tempfile
    with tempfile.NamedTemporaryFile(prefix='digit-ae-sudoers-', suffix='.conf') as f:
        f.write(data); f.flush()
        command(['/usr/sbin/visudo', '-cf', f.name])

def restore():
    idle()
    state = read(ADMIN/'installation.json')
    require(state['state'] in ('installing', 'installed', 'restored'), 'Unknown deployment state')
    if state['state'] == 'restored':
        return
    for name, row in state['targets'].items():
        p = Path(name)
        require(not p.is_symlink(), 'Target symlink: ' + name)
        allowed = {row['after_sha256']}
        if row['before_sha256']:
            require(sha(ADMIN/row['backup']) == row['before_sha256'], 'Backup changed: ' + name)
            allowed.add(row['before_sha256'])
        require((not p.exists() and row['before_sha256'] is None) or (p.exists() and sha(p) in allowed), 'Target changed since install: ' + name)
    for name, row in state['targets'].items():
        p = Path(name)
        if row['before_sha256']:
            atomic(p, (ADMIN/row['backup']).read_bytes(), row['before_mode'])
            os.chown(p, row['before_uid'], row['before_gid'])
        elif p.exists():
            p.unlink()
    command(['/usr/bin/systemctl', 'daemon-reload'])
    state['state'] = 'restored'
    write(ADMIN/'installation.json', state)

def checks():
    state = read(ADMIN/'installation.json')
    for name, row in state['targets'].items():
        require(sha(name) == row['after_sha256'], 'Installed target drift: ' + name)
    for name, digest in read(ADMIN/'payload.json')['files'].items():
        require(sha(ADMIN/name) == digest, 'Installed package drift: ' + name)
    command(['/usr/sbin/visudo', '-cf', str(SUDOERS)])
    for d in DATASETS:
        for m in MODES:
            require(command(['/usr/bin/systemctl', 'show', unit(d, m), '-p', 'LoadState', '--value']).strip() == 'loaded', 'Unit failed to load')
    command(['/usr/sbin/runuser', '-u', 'atc27_ae', '--', str(CLI), 'unified'])
    command(['/usr/sbin/runuser', '-u', 'atc27_ae', '--', str(CLI), '--help'])
    return dict(passed=True, unit_definitions=15, launched_experiments=0)

def install():
    manifest = payload()
    idle()
    if (ADMIN/'installation.json').exists():
        require(read(ADMIN/'installation.json')['state'] == 'installed', 'Inspect previous deployment before reusing')
        print(__import__('json').dumps(checks(), indent=2)); return
    require(not ADMIN.exists(), 'Preserve existing deployment directory')
    require(not CLI.is_symlink() and sha(CLI) == GATE_SHA, 'Existing entry differs from reviewed pause gate')
    values = targets()
    for name in values:
        if name != str(CLI):
            require(not Path(name).exists() and not Path(name).is_symlink(), 'Existing new target: ' + name)
    check_sudoers(sudoers_text())
    subprocess.run(['/usr/bin/python3', '-I', '-B', str(HERE/'tests.py')], check=True, timeout=60)
    ADMIN.mkdir(mode=0o755)
    (ADMIN/'before').mkdir(mode=0o700)
    (ADMIN/'enabled').mkdir(mode=0o755)
    (ADMIN/'bindings').mkdir(mode=0o755)
    OUT.mkdir(parents=True, exist_ok=True, mode=0o755)
    for name in (*FILES, 'payload.json'):
        atomic(ADMIN/name, (HERE/name).read_bytes())
    atomic(ADMIN/'legacy_gate.py', CLI.read_bytes())
    write(ADMIN/'registry.json', dict(schema='digit-ae-unified-registry-v1', datasets={d: dict(enabled=False,
        reason='Awaiting short-window acceptance of the bound unified source version') for d in DATASETS}))
    state = dict(state='installing', payload=manifest, targets={})
    for name, row in values.items():
        p = Path(name)
        st = p.stat() if p.exists() else None
        backup = 'before/' + p.name
        if st:
            atomic(ADMIN/backup, p.read_bytes(), st.st_mode & 0o7777)
            require(sha(ADMIN/backup) == sha(p), 'Backup failed')
        state['targets'][name] = dict(after_sha256=hashlib.sha256(row['data']).hexdigest(),
            before_sha256=sha(p) if st else None, backup=backup,
            before_mode=st.st_mode & 0o7777 if st else None,
            before_uid=st.st_uid if st else None, before_gid=st.st_gid if st else None)
    write(ADMIN/'installation.json', state)
    try:
        for name, row in values.items():
            atomic(name, row['data'], row['mode'])
        command(['/usr/bin/systemctl', 'daemon-reload'])
        state['verification'] = checks()
        state['state'] = 'installed'
        write(ADMIN/'installation.json', state)
        print(__import__('json').dumps(state['verification'], indent=2))
    except BaseException:
        restore()
        raise

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--restore', action='store_true')
    a = p.parse_args()
    payload()
    if not a.execute:
        print(__import__('json').dumps(dict(action='restore' if a.restore else 'install',
            destination=str(ADMIN), units=[unit(d, m) for d in DATASETS for m in MODES],
            initial_state='all_pending_acceptance', starts_experiments=False,
            original_pa_main_unchanged=True, old_launch_gates_retained=True), indent=2))
        return
    require(os.geteuid() == 0, 'Run locally with sudo to install root-owned CLI, service units and sudoers')
    if a.restore:
        restore()
    else:
        install()

if __name__ == '__main__':
    main()
