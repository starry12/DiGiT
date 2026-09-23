#!/usr/bin/env python3
"""Independent DiGiT AE package entry: new experiments and archived references."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Direct python invocations must also keep the immutable package cache-free.
sys.dont_write_bytecode = True
# This import resolves beside this file; no original-workspace code is loaded.
from artifact_integrity import ROOT, checked_static_path, read_json, sha, verify_package

ACTIONS = ('verify', 'matrix', 'environment', 'example', 'check', 'smoke',
           'representative', 'summarize', 'reference')
NATIVE = frozenset(('check', 'smoke', 'representative'))
NO_OUTPUT = frozenset(('verify', 'matrix', 'environment', 'reference'))
DATA_MOUNTS = (
    'data', 'ssd_state',
    'results/pa_sage_direction_20260921_v2/bidirectional/valid_trace',
    'results/pa_sage_direction_20260921_v2/bidirectional/test_trace',
    'results/pa_sage_g2_ssd_20260920_v2/ssd_verify_receipt.json',
)


def readonly_mount(path):
    import re
    target = str(Path(path))
    def decode(value):
        return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        if len(fields) > 6 and decode(fields[4]) == target:
            return 'ro' in fields[5].split(',') and bool(os.statvfs(target).f_flag & os.ST_RDONLY)
    return False


def data_identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, mode=info.st_mode,
                bytes=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def _result_path(value, must_exist=False):
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    # Reject traversal and symlinks before normalization, including the results
    # directory itself. Runtime output must never write through a data mount.
    if '..' in candidate.parts:
        raise ValueError('Result paths must not contain ..')
    try:
        relative = candidate.relative_to(ROOT)
    except ValueError:
        raise ValueError('Use a directory inside this package results/')
    if len(relative.parts) < 2 or relative.parts[0] != 'results':
        raise ValueError('Use a fresh directory below this package results/')
    current = ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Result paths must not traverse symlinks: ' + str(current))
    if must_exist:
        if not candidate.is_dir():
            raise ValueError('Input result directory does not exist: ' + str(candidate))
    elif candidate.exists():
        raise ValueError('Output already exists; select a fresh results/ directory: ' + str(candidate))
    return candidate


def _deployment(package_sha):
    directory = ROOT / 'deployment'
    path = directory / 'local.json'
    if directory.is_symlink() or path.is_symlink():
        raise RuntimeError('Deployment binding must be a local regular file; see docs/DATA.md')
    if not path.is_file():
        raise RuntimeError('Native execution is not configured: deployment/local.json is missing. Follow docs/DATA.md to bind data and SSD verification evidence to this package.')
    document = read_json(path)
    if not isinstance(document, dict) or document.get('package_sha256') != package_sha:
        raise RuntimeError('Deployment binding does not match this package. Follow docs/DATA.md; preparation and SSD readiness are not implied by a copied configuration.')
    if document.get('schema') != 'digit-ae-deployment-v1':
        raise RuntimeError('Unsupported deployment schema; see docs/DATA.md')
    locations = document.get('locations')
    if not isinstance(locations, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in locations.items()):
        raise RuntimeError('Deployment locations must be a string mapping')
    mounts = document.get('mounts')
    if not isinstance(mounts, list) or len(mounts) != len(DATA_MOUNTS):
        raise RuntimeError('Deployment must register exactly the documented data mounts')
    seen = set()
    for mount in mounts:
        if not isinstance(mount, dict) or mount.get('path') not in DATA_MOUNTS or mount['path'] in seen:
            raise RuntimeError('Unapproved or duplicate deployment mount; code and whole-results mounts are forbidden')
        name = mount['path']
        seen.add(name)
        target_value = mount.get('target')
        if not isinstance(target_value, str):
            raise RuntimeError('Deployment target is not an absolute path: ' + name)
        target = Path(target_value)
        if not target.is_absolute() or str(target.resolve(strict=True)) != target_value:
            raise RuntimeError('Deployment target must be a canonical absolute path: ' + name)
        destination = ROOT / name
        current = ROOT
        for part in Path(name).parts[:-1]:
            current = current / part
            if current.is_symlink() or not current.is_dir():
                raise RuntimeError('Deployment mount parent is not a local directory: ' + str(current))
        mode = mount.get('binding_mode','symlink')
        if mode == 'symlink':
            if not destination.is_symlink() or os.readlink(str(destination)) != target_value:
                raise RuntimeError('Deployment data link changed or is missing: ' + name)
        elif mode == 'pre_mounted':
            if destination.is_symlink() or target_value != str(destination):
                raise RuntimeError('Pre-mounted data must use the local destination: ' + name)
            if not readonly_mount(destination):
                raise RuntimeError('Registered data is no longer mounted: ' + name)
            if not (os.statvfs(str(destination)).f_flag & os.ST_RDONLY):
                raise RuntimeError('Registered data mount is not read-only: ' + name)
        else:
            raise RuntimeError('Unknown data binding mode: ' + name)
        if data_identity(destination) != mount.get('identity'):
            raise RuntimeError('Prepared data identity changed: ' + name + '; repeat preparation checks before rebinding')
        expected_file = name.endswith('.json')
        if (expected_file and not destination.is_file()) or (not expected_file and not destination.is_dir()):
            raise RuntimeError('Deployment target has the wrong type: ' + name)
    if seen != set(DATA_MOUNTS):
        raise RuntimeError('Deployment data mounts are incomplete')
    return document


def _new_result(source, package_sha, dataset, model, actions=('smoke', 'representative')):
    receipt = source / 'artifact_invocation.json'
    if receipt.is_symlink() or not receipt.is_file():
        raise RuntimeError('Input has no new-package invocation receipt. Use reference to inspect archived results; run new experiments through this package run.sh.')
    document = read_json(receipt)
    if (not isinstance(document, dict) or
            document.get('schema') != 'digit-ae-artifact-invocation-v1' or
            document.get('package_sha256') != package_sha or
            document.get('action') not in actions or
            document.get('exit_code') != 0 or
            document.get('dataset') != dataset or document.get('model') != model or
            document.get('output') != source.relative_to(ROOT).as_posix()):
        raise RuntimeError('Input is not a successful, matching run of this package; archived evidence is available through reference')


def _command(args, output, source):
    if args.action == 'example':
        return [sys.executable, '-B', str(ROOT/'tools/cpu_example.py'), '--output', str(output)]
    if args.action == 'environment':
        return [sys.executable, '-B', str(ROOT/'environment/check.py')]
    prefix = 'evaluation.sage' if args.model == 'sage' else 'training.' + args.model
    if args.action == 'check':
        return [sys.executable, '-u', '-m', prefix+'.check', '--output', str(output), '--gpu', str(args.gpu)]
    if args.action == 'summarize':
        return [sys.executable, '-u', '-m', prefix+'.summarize', '--input', str(source), '--output', str(output)]
    return [sys.executable, '-u', '-m', prefix+'.controller', '--mode', args.action, '--output', str(output), '--gpu', str(args.gpu)]


def _environment():
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONOPTIMIZE', None)
    environment.pop('DIGIT_ARTIFACT_PREFLIGHT', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    # Never inherit the original checkout's library search path.
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(
        str(ROOT / path) for path in ('third_party/bam/build/lib', 'runtime/lib')
        if (ROOT / path).is_dir())
    return environment


def _reference():
    return dict(historical_evidence=True, new_execution=False, results=read_json(checked_static_path('reference/results.json')))


def _write_receipt(output, package_sha, args, command, code, started):
    if output is None or not output.is_dir():
        return
    _result_path(output, must_exist=True)
    receipt = output / 'artifact_invocation.json'
    if receipt.exists() or receipt.is_symlink():
        raise RuntimeError('Refusing to replace an existing invocation receipt: ' + str(receipt))
    document = dict(schema='digit-ae-artifact-invocation-v1', package_sha256=package_sha,
                    action=args.action, dataset=args.dataset, model=args.model,
                    output=output.relative_to(ROOT).as_posix(), command=command,
                    exit_code=code, started_unix=started, finished_unix=time.time())
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write('\n')



def _preflight_file(folder, name):
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Unsafe preflight evidence path')
    path = folder
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise RuntimeError('Preflight evidence must not be a symlink: ' + str(path))
    if not path.is_file():
        raise RuntimeError('Preflight evidence is missing: ' + str(path))
    return path


def _check_status(folder):
    status = read_json(_preflight_file(folder, 'status.json'))
    if not isinstance(status, dict) or status.get('passed') is not True or status.get('complete') is not True:
        raise RuntimeError('This package must complete its own check before native execution')
    evidence = status.get('evidence_sha256', {})
    if not isinstance(evidence, dict):
        raise RuntimeError('Malformed preflight evidence mapping')
    for name, digest in evidence.items():
        if sha(_preflight_file(folder, name)) != digest:
            raise RuntimeError('Preflight evidence changed: ' + name)
    if 'input_binding_sha256' in status and sha(_preflight_file(folder, 'inputs.json')) != status['input_binding_sha256']:
        raise RuntimeError('Preflight input binding changed')
    return status


def _preflight_receipt(args):
    folder = ROOT / 'deployment/checks'
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise RuntimeError('deployment/checks must be a local directory')
    path = folder / (args.dataset + '_' + args.model + '.json')
    if path.is_symlink():
        raise RuntimeError('Preflight receipt must not be a symlink')
    return path


def _record_preflight(output, package_sha, args):
    _new_result(output, package_sha, args.dataset, args.model, actions=('check',))
    _check_status(output)
    receipt = _preflight_receipt(args)
    receipt.parent.mkdir(exist_ok=True)
    value = dict(schema='digit-ae-package-preflight-v1', package_sha256=package_sha,
                 dataset=args.dataset, model=args.model, gpu=args.gpu,
                 output=output.relative_to(ROOT).as_posix(),
                 status_sha256=sha(output / 'status.json'),
                 invocation_sha256=sha(output / 'artifact_invocation.json'))
    temporary = receipt.with_name(receipt.name + '.tmp.' + str(os.getpid()))
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(str(temporary), str(receipt))
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _preflight(args, package_sha):
    receipt = _preflight_receipt(args)
    if not receipt.is_file():
        raise RuntimeError('No completed package check for ' + args.dataset + '/' + args.model + '. Run check with this package first; old-workspace preflight results are not accepted. See docs/DATA.md.')
    value = read_json(receipt)
    if (not isinstance(value, dict) or value.get('schema') != 'digit-ae-package-preflight-v1' or
            value.get('package_sha256') != package_sha or value.get('dataset') != args.dataset or
            value.get('model') != args.model or not isinstance(value.get('output'), str)):
        raise RuntimeError('Preflight receipt does not match the current package and selected pair')
    output = _result_path(value['output'], must_exist=True)
    _new_result(output, package_sha, args.dataset, args.model, actions=('check',))
    if (sha(_preflight_file(output, 'status.json')) != value.get('status_sha256') or
            sha(_preflight_file(output, 'artifact_invocation.json')) != value.get('invocation_sha256')):
        raise RuntimeError('Package preflight receipt or status changed')
    _check_status(output)
    if value.get('gpu') != args.gpu:
        raise RuntimeError('Preflight GPU differs from requested physical GPU; run a matching check')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=ACTIONS)
    parser.add_argument('--dataset', choices=('PA',), default='PA')
    parser.add_argument('--model', choices=('sage', 'gcn', 'gat'), default='sage')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--accepted', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.accepted:
        parser.error('--accepted is disabled. Use reference for archived evidence or summarize --input for a new package run.')
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be a nonnegative physical GPU index')
    if args.action in NO_OUTPUT and args.output is not None:
        parser.error('This action does not create an output directory')
    if args.action not in NO_OUTPUT and args.output is None:
        parser.error('--output is required; choose a fresh directory below results/')
    if args.action in NATIVE and args.gpu is None:
        parser.error('Choose an idle physical --gpu')
    if args.action == 'summarize' and args.input is None:
        parser.error('summarize requires --input from a new package experiment')
    if args.action != 'summarize' and args.input is not None:
        parser.error('--input is only valid with summarize')
    package_sha = verify_package()
    output = _result_path(args.output) if args.output is not None else None
    source = _result_path(args.input, must_exist=True) if args.input is not None else None
    if source is not None:
        _new_result(source, package_sha, args.dataset, args.model)
    if args.action == 'verify':
        print(json.dumps(dict(verified=True, package_root=str(ROOT), package_sha256=package_sha, deployment_readiness_claim=False), indent=2))
        return 0
    if args.action == 'reference':
        print(json.dumps(_reference(), indent=2))
        return 0
    if args.action == 'matrix':
        print(json.dumps(dict(dataset='PA', models=['sage','gcn','gat'], systems=['GIDS','DiGiT'], seed=0, epochs=20, bfs=False), indent=2))
        return 0
    command = _command(args, output, source)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, command=command, cwd=str(ROOT), package_sha256=package_sha,
                              dataset=args.dataset, model=args.model, gpu=args.gpu,
                              output_created=False, deployment_checked=False, raw_ssd_writes=False,
                              epochs=20,
                              smoke_before_full=args.action == 'representative'), indent=2))
        return 0
    environment = _environment()
    if args.action in NATIVE:
        _deployment(package_sha)
    if args.action in ('smoke', 'representative'):
        environment['DIGIT_ARTIFACT_PREFLIGHT'] = str(_preflight(args, package_sha))
    started = time.time()
    child = subprocess.Popen(command, cwd=str(ROOT), env=environment)
    try:
        code = child.wait()
    except KeyboardInterrupt:
        import signal
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
        finally:
            signal.signal(signal.SIGINT, previous)
        code = 130
    # The child owns output creation and its native acceptance checks. The
    # wrapper only records provenance after it exits, including failed runs.
    _write_receipt(output, package_sha, args, command, code, started)
    if args.action == 'check' and code == 0:
        if verify_package() != package_sha:
            raise RuntimeError('Package changed while the check was running')
        _deployment(package_sha)
        _record_preflight(output, package_sha, args)
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
