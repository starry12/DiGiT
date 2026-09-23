#!/usr/bin/env python3
"""Record package metadata and compiler versions without importing GPU libraries."""
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / 'environment'
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
VERSION = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+!-]*$')


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def os_release():
    wanted = {'NAME', 'VERSION', 'ID', 'VERSION_ID', 'PRETTY_NAME'}
    values = {}
    path = Path('/etc/os-release')
    if path.is_file():
        for line in path.read_text(errors='replace').splitlines():
            key, separator, value = line.partition('=')
            if separator and key in wanted:
                values[key] = value.strip('"\'')
    return values


def compiler(variable, default):
    requested = os.environ.get(variable, default)
    binary = shutil.which(requested)
    if not binary:
        return {'available': False}
    try:
        # Only compiler version commands are executed; never query the GPU driver.
        result = subprocess.run([binary, '--version'], capture_output=True,
                                text=True, timeout=5, check=False)
        return {'available': True, 'returncode': result.returncode,
                'version_output': (result.stdout + result.stderr).strip()[:4096]}
    except (OSError, subprocess.SubprocessError) as error:
        return {'available': True, 'error_type': type(error).__name__}


def conda_packages():
    result = []
    directory = Path(sys.prefix) / 'conda-meta'
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob('*.json')):
        try:
            record = json.loads(path.read_text())
            name = str(record.get('name', ''))
            version = str(record.get('version', ''))
            if NAME.fullmatch(name) and VERSION.fullmatch(version):
                # Deliberately omit channels, source URLs, prefixes and credentials.
                result.append({'name': name, 'version': version,
                               'build': str(record.get('build', '')),
                               'subdir': str(record.get('subdir', ''))})
        except (OSError, ValueError):
            result.append({'metadata_unreadable': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True,
                        help='New directory under this artifact results/; relative paths use the artifact root')
    args = parser.parse_args()
    requested = Path(args.output).expanduser()
    if not requested.is_absolute():
        requested = ROOT / requested
    # Reject symlink components before resolving, including dangling output links.
    for path in [requested] + list(requested.parents):
        if path == ROOT.parent:
            break
        if path.is_symlink():
            parser.error('output path must not traverse symlinks')
    output = requested.resolve()
    allowed = ROOT / 'results'
    if output == allowed or allowed not in output.parents:
        parser.error('--output must be a new subdirectory of this artifact results/')
    if output.exists():
        parser.error('--output must not already exist')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    packages = []
    skipped = 0
    for distribution in metadata.distributions():
        name = str(distribution.metadata.get('Name', ''))
        version = str(distribution.version)
        if NAME.fullmatch(name) and VERSION.fullmatch(version):
            packages.append({'name': name, 'version': version})
        else:
            skipped += 1
    packages.sort(key=lambda package: (normalized(package['name']), package['version']))
    observed_versions = {}
    for package in packages:
        observed_versions.setdefault(normalized(package['name']), set()).add(package['version'])

    comparison = {'available': False, 'scope': 'Installed metadata versions only; no imports or runtime checks'}
    expected_path = LOCKS / 'observed.json'
    if expected_path.is_file():
        expected = json.loads(expected_path.read_text())
        excluded = {normalized(name) for name in expected.get('excluded_local_packages', [])}
        differences = []
        expected_python = str(expected.get('python', '')).split()[0]
        if platform.python_version() != expected_python:
            differences.append({'package': 'Python', 'expected': expected_python,
                                'observed': platform.python_version()})
        for package in expected.get('packages', []):
            key = normalized(package['name'])
            if key in excluded:
                continue
            installed = sorted(observed_versions.get(key, set()))
            if installed != [package['version']]:
                differences.append({'package': package['name'], 'expected': package['version'],
                                    'observed': installed})
        comparison.update(available=True, observed_versions_match=not differences, differences=differences,
                          reference='environment/observed.json',
                          reference_sha256=sha(expected_path))

    driver = None
    driver_path = Path('/proc/driver/nvidia/version')
    if driver_path.is_file():
        match = re.search(r'Kernel Module\s+([0-9.]+)', driver_path.read_text(errors='replace'))
        if match:
            driver = match.group(1)
    locks = {}
    for name in ('conda-linux-64.lock', 'requirements.lock', 'wheels.json', 'observed.json'):
        path = LOCKS / name
        if path.is_file():
            locks['environment/' + name] = sha(path)
    report = {
        'schema': 'digit-ae-environment-snapshot-v1',
        'recorded_utc': datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        'os': os_release(), 'kernel': platform.release(), 'architecture': platform.machine(),
        'python': {'version': platform.python_version(), 'implementation': platform.python_implementation(),
                   'compiler': platform.python_compiler()},
        'compilers': {'nvcc': compiler('DIGIT_NVCC', 'nvcc'),
                      'cmake': compiler('DIGIT_CMAKE', 'cmake'), 'cc': compiler('CC', 'cc')},
        'nvidia_driver_version_from_proc': driver,
        'packages': packages, 'package_records_skipped': skipped, 'conda_packages': conda_packages(),
        'reference_locks_sha256': locks, 'version_comparison': comparison,
        'scope': 'OS/kernel/compiler/package metadata snapshot; not an environment or native acceptance test',
        'python_gpu_libraries_imported': False, 'gpu_commands_executed': False,
        'ssd_access_performed': False, 'clean_install_proven': False, 'native_acceptance_proven': False,
        'observed_list_is_install_lock': False,
    }
    (output / 'environment.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    header = ('# Observed installed package names and versions only.\n'
              '# This is not an install lock and does not prove a clean installation.\n'
              '# No source URLs, index settings, editable paths or credentials are recorded.\n')
    (output / 'pip_observed.txt').write_text(header + ''.join(
        package['name'] + '==' + package['version'] + '\n' for package in packages))
    print(json.dumps({'snapshot_written': True, 'output': str(output.relative_to(ROOT)),
                      'packages': len(packages), 'observed_versions_match': comparison.get('observed_versions_match'),
                      'clean_install_proven': False, 'native_acceptance_proven': False}, indent=2))


if __name__ == '__main__':
    main()
