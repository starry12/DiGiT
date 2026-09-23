"""Independent, uncached integrity checks for this extracted artifact package.

Only standard-library modules are used. This module never locates or imports a
research workspace, opens a GPU, or touches an SSD device.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'ARTIFACT_MANIFEST.json'
# Deployment data are checked by the deployment/native data checks, not treated
# as immutable source code. These directories are absent from a clean package.
MUTABLE_ROOTS = frozenset(('results', 'deployment', 'data', 'ssd_state', '.git', 'build', '.native-build'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key: ' + key)
        value[key] = item
    return value


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Manifest path must be a nonempty POSIX relative path')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or name == '.':
        raise ValueError('Unsafe manifest path: ' + repr(name))
    if path.parts[0] in MUTABLE_ROOTS or name == MANIFEST.name:
        raise ValueError('Mutable or self-referential manifest entry: ' + name)
    return path


def checked_static_path(name):
    relative = _relative(name)
    current = ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError('Static package symlink is forbidden: ' + name)
    if not current.is_file():
        raise RuntimeError('Package file is missing: ' + name)
    return current


def _manifest():
    if MANIFEST.is_symlink():
        raise RuntimeError('ARTIFACT_MANIFEST.json must not be a symlink')
    if not MANIFEST.is_file():
        raise RuntimeError('Package preparation is incomplete: ARTIFACT_MANIFEST.json is missing')
    document = read_json(MANIFEST)
    if not isinstance(document, dict) or not isinstance(document.get('files'), dict) or not document['files']:
        raise ValueError('ARTIFACT_MANIFEST.json requires a nonempty files mapping')
    return document


def verify_package():
    """Hash every immutable file and reject unlisted files or code symlinks."""
    before = sha(MANIFEST) if MANIFEST.is_file() and not MANIFEST.is_symlink() else None
    document = _manifest()
    expected = document['files']
    for name, digest in expected.items():
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid SHA-256 for manifest path: ' + str(name))
        path = checked_static_path(name)
        if sha(path) != digest:
            raise RuntimeError('Package file changed: ' + name)
    found = set()
    for base, directories, files in os.walk(str(ROOT), followlinks=False):
        relative = Path(base).relative_to(ROOT)
        kept = []
        for name in directories:
            if relative == Path('.') and name in MUTABLE_ROOTS:
                continue
            path = Path(base) / name
            if path.is_symlink():
                raise RuntimeError('Static package symlink is forbidden: ' + path.relative_to(ROOT).as_posix())
            kept.append(name)
        directories[:] = kept
        for name in files:
            path = Path(base) / name
            key = path.relative_to(ROOT).as_posix()
            if key == MANIFEST.name:
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError('Static package file must be a regular file: ' + key)
            found.add(key)
    extra = found.difference(expected)
    if extra:
        raise RuntimeError('Unlisted static package files: ' + ', '.join(sorted(extra)[:8]))
    if found != set(expected):
        raise RuntimeError('Manifest files changed during verification')
    after = sha(MANIFEST)
    if before != after:
        raise RuntimeError('Package manifest changed during verification')
    return after


def verify_component(component_dir):
    """Verify the entire package first, then return its component receipt hash."""
    verify_package()
    component = Path(component_dir)
    if not component.is_absolute():
        component = ROOT / component
    try:
        relative = component.relative_to(ROOT)
    except ValueError:
        raise RuntimeError('Component is outside this package: ' + str(component))
    name = (relative / 'manifest.json').as_posix()
    path = checked_static_path(name)
    if name not in _manifest()['files']:
        raise RuntimeError('Component manifest is not bound by the package: ' + name)
    return sha(path)
