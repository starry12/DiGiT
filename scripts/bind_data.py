#!/usr/bin/env python3
"""Bind an existing prepared dataset tree; never modify it or access raw SSDs."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from artifact_integrity import ROOT, read_json, verify_package
from artifact import DATA_MOUNTS, _deployment, data_identity, readonly_mount


def read_locations(argument, prepared):
    if argument is None:
        source = prepared / 'configs/external_paths.json'
        if not source.is_file():
            raise RuntimeError('Prepared profile has no configs/external_paths.json; supply --locations JSON')
        value = read_json(source)
    elif argument.lstrip().startswith('{'):
        value = json.loads(argument)
    else:
        source = Path(argument).expanduser()
        if not source.is_file():
            raise RuntimeError('--locations must be a JSON object or the path to a JSON file')
        value = read_json(source)
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise ValueError('Locations must be a JSON string-to-string mapping')
    return value


def destination_available(name):
    path = ROOT / name
    current = ROOT
    for part in Path(name).parts[:-1]:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise RuntimeError('Refusing a non-local destination parent: ' + str(current))
    if path.exists() or path.is_symlink():
        raise RuntimeError('Refusing to overwrite an existing binding target: ' + str(path))
    return path


def make_plan(prepared, locations):
    package_sha = verify_package()
    prepared = prepared.expanduser().resolve(strict=True)
    if not prepared.is_dir():
        raise ValueError('--prepared-root must be an existing prepared project directory')
    if prepared == ROOT or ROOT in prepared.parents:
        raise ValueError('Prepared data must be outside this package')
    binding = ROOT / 'deployment/local.json'
    if binding.exists() or binding.is_symlink():
        raise RuntimeError('deployment/local.json already exists; binding never overwrites a deployment')
    destination_available('deployment/local.json')
    mounts = []
    for name in DATA_MOUNTS:
        destination_available(name)
        source = prepared / name
        if not source.exists():
            raise RuntimeError('Prepared profile is incomplete; source is missing: ' + str(source))
        source = source.resolve(strict=True)
        expected_file = name.endswith('.json')
        if (expected_file and not source.is_file()) or (not expected_file and not source.is_dir()):
            raise RuntimeError('Prepared data target has the wrong type: ' + str(source))
        mounts.append(dict(path=name, target=str(source), identity=data_identity(source)))
    return dict(schema='digit-ae-deployment-v1', package_sha256=package_sha,
                prepared_root=str(prepared), locations=read_locations(locations, prepared),
                mounts=mounts, read_only_data_use=True, raw_ssd_access=False,
                native_readiness_claim=False, created_unix=time.time())


def bind(plan):
    created_directories = []
    created_links = []
    created_binding = False
    lock = None
    lock_created = False
    binding = ROOT / 'deployment/local.json'

    def ensure_directory(path):
        relative = path.relative_to(ROOT)
        current = ROOT
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise RuntimeError('Refusing a non-local destination parent: ' + str(current))
            if not current.exists():
                current.mkdir()
                created_directories.append(current)

    try:
        # All five sources and destinations were inspected before the first write.
        for mount in plan['mounts']:
            destination_available(mount['path'])
            if data_identity(mount['target']) != mount['identity']:
                raise RuntimeError('Prepared source changed before binding: ' + mount['path'])
        ensure_directory(binding.parent)
        lock = binding.parent / 'bind.lock'
        descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        lock_created = True
        for mount in plan['mounts']:
            destination = destination_available(mount['path'])
            ensure_directory(destination.parent)
            os.symlink(mount['target'], str(destination), target_is_directory=not mount['path'].endswith('.json'))
            created_links.append(destination)
        if verify_package() != plan['package_sha256']:
            raise RuntimeError('Package changed during binding')
        with binding.open('x', encoding='utf-8') as stream:
            created_binding = True
            json.dump(plan, stream, indent=2, sort_keys=True)
            stream.write('\n')
        _deployment(plan['package_sha256'])
    except BaseException:
        if created_binding:
            binding.unlink()
        for path in reversed(created_links):
            if path.is_symlink():
                path.unlink()
        raise
    finally:
        if lock_created:
            lock.unlink()
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except OSError:
                pass
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True,
                        help='Existing prepared project data profile; no source code is imported')
    parser.add_argument('--locations', metavar='JSON',
                        help='Override locations with a JSON object or JSON-file path')
    parser.add_argument('--pre-mounted', action='store_true', help='Register existing administrator-provided read-only mounts; create no symlinks or mounts')
    parser.add_argument('--dry-run', action='store_true', help='Inspect sources and print the plan without creating anything')
    args = parser.parse_args(argv)
    if args.pre_mounted:
        package_sha = verify_package()
        prepared = args.prepared_root.expanduser().resolve(strict=True)
        destination_available('deployment/local.json')
        mounts = []
        for name in DATA_MOUNTS:
            path = ROOT / name
            current = ROOT
            for part in Path(name).parts:
                current = current / part
                if current.is_symlink():
                    raise RuntimeError('Pre-mounted path must not traverse symlinks: '+str(current))
            if not path.exists() or not readonly_mount(path):
                raise RuntimeError('Expected administrator-mounted path: '+str(path))
            if not (os.statvfs(str(path)).f_flag & os.ST_RDONLY):
                raise RuntimeError('Data mount must be read-only: '+str(path))
            if (name.endswith('.json') and not path.is_file()) or (not name.endswith('.json') and not path.is_dir()):
                raise RuntimeError('Wrong mounted object type: '+name)
            mounts.append(dict(path=name,target=str(path),identity=data_identity(path),binding_mode='pre_mounted'))
        plan=dict(schema='digit-ae-deployment-v1',package_sha256=package_sha,
                  prepared_root=str(prepared),locations=read_locations(args.locations,prepared),
                  mounts=mounts,read_only_data_use=True,raw_ssd_access=False,
                  native_readiness_claim=False,created_unix=time.time())
        if not args.dry_run:
            directory=ROOT/'deployment';directory.mkdir(exist_ok=True)
            with (directory/'local.json').open('x') as stream:
                json.dump(plan,stream,indent=2,sort_keys=True);stream.write('\n')
            try: _deployment(package_sha)
            except BaseException:
                (directory/'local.json').unlink();raise
    else:
        plan = make_plan(args.prepared_root, args.locations)
    if not args.dry_run and not args.pre_mounted:
        bind(plan)
    print(json.dumps(dict(dry_run=args.dry_run, deployment_written=not args.dry_run,
                          native_readiness_claim=False, raw_ssd_access=False, deployment=plan), indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
