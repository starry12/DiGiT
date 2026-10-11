#!/usr/bin/env python3
"""Archive an exact AE candidate; a snapshot is not proof of paper provenance."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(snapshot, live_root=None):
    snapshot = Path(snapshot)
    manifest_path = snapshot / 'manifest.json'
    if digest(manifest_path) != (snapshot / 'manifest.sha256').read_text().split()[0]:
        raise RuntimeError('Snapshot manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    archive = snapshot / 'sources.tar.gz'
    if digest(archive) != manifest['archive_sha256']:
        raise RuntimeError('Snapshot archive checksum mismatch')
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            if (not member.isfile() or name in seen or name not in manifest['files']
                    or Path(name).is_absolute() or '..' in Path(name).parts):
                raise RuntimeError('Unexpected archive member: ' + name)
            expected = manifest['files'][name]
            h = hashlib.sha256()
            with tar.extractfile(member) as f:
                for chunk in iter(lambda: f.read(8 * 1024**2), b''):
                    h.update(chunk)
            if member.size != expected['bytes'] or h.hexdigest() != expected['sha256']:
                raise RuntimeError('Archived file changed: ' + name)
            if live_root is not None:
                path = Path(live_root) / name
                if not path.is_file() or digest(path) != expected['sha256']:
                    raise RuntimeError('Live file differs from snapshot: ' + name)
            seen.add(name)
    if seen != set(manifest['files']):
        raise RuntimeError('Missing archive members')
    return {'passed': True, 'snapshot_id': manifest['snapshot_id'],
            'manifest_sha256': digest(manifest_path), 'files': len(seen),
            'live_files_checked': live_root is not None,
            'paper_original_version': False, 'full_data_hashes_checked': False}


def create(snapshot, protocol=None, role="paper_reconstruction_starting_implementation_and_protocol"):
    protocol = Path(protocol) if protocol is not None else ROOT / "configs/paper/pa_sage.json"
    if not protocol.is_absolute():
        protocol = ROOT / protocol
    import sys
    sys.path.insert(0, str(ROOT))
    from ae.common import verify_release
    release = verify_release()
    snapshot = Path(snapshot).resolve()
    snapshot.mkdir(parents=True, exist_ok=False)
    release_files = json.loads((ROOT / 'ae/release_manifest.json').read_text())['files']
    names = set(release_files)
    names.update(['ae/release_manifest.json', '.digit-root', 'atc26-paper1508.pdf',
                  'AE_SCOPE.md', 'docs/ae_scope.json', 'docs/PA_SAGE_FREEZE.md',
                  'docs/pa_sage_provenance.json'])
    # Include preprocessing and existing correctness tests, excluding generated files.
    for path in (ROOT / 'evaluation/digit').rglob('*'):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
            names.add(str(path.relative_to(ROOT)))
    for relative in ('README.md', 'docs/PA_SAGE_RANDOM_V2.md', 'docs/PA_SAGE_NATIVE_V3.md', 'docs/PA_SAGE_FORMAL_V4.md'):
        if (ROOT/relative).is_file():
            names.add(relative)
    execution_binding = None
    if role == 'pa_sage_formal_execution_v4':
        from ae.pa_sage.common import DEFAULT_DATA, read
        from ae.pa_sage.native_gate import current_binding
        execution_binding = current_binding(DEFAULT_DATA)
        ready_path = DEFAULT_DATA/'ssd_ready.json'
        ready = read(ready_path)
        baseline = read(ROOT/'configs/ssd_payloads.json')['papers_gids']
        # Preserve the essential SSD evidence at its current referenced path.
        # Bulk input data and old experiment logs are not copied.
        for path in (ready_path, Path(ready['state']), Path(ready['verify_receipt']), Path(baseline['state'])):
            names.add(str(path.resolve().relative_to(ROOT)))
    files = {}
    for name in sorted(names):
        path = ROOT / name
        files[name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    archive = snapshot / 'sources.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in sorted(files):
            path = ROOT / name
            # Store bytes, never a live link into mutable source/data directories.
            info = tarfile.TarInfo(name)
            info.size = files[name]['bytes']
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            info.mtime = 0
            with path.open('rb') as f:
                tar.addfile(info, f)
    manifest = {
        'schema': 'digit-ae-source-snapshot-v1', 'snapshot_id': snapshot.name,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'role': role,
        'paper_original_version': False, 'paper_aligned_pa_training_ready': False,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        'git_head_is_complete_version': False,
        'release_sha256': release,
        'execution_binding': execution_binding,
        'paper_sha256': digest(ROOT / 'atc26-paper1508.pdf'),
        'protocol_path': str(protocol.relative_to(ROOT)),
        'protocol_sha256': digest(protocol),
        'archive_sha256': digest(archive), 'files': files,
        'data_binding': 'Data manifests and SSD receipts only; bulk inputs remain external. Their stored hashes were not all recomputed.',
        'binary_build_provenance': 'Existing binaries captured by hash; clean rebuild and source-to-binary correspondence not yet verified.',
    }
    (snapshot / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (snapshot / 'manifest.sha256').write_text(digest(snapshot / 'manifest.json') + '  manifest.json\n')
    result = verify(snapshot, ROOT)
    (snapshot / 'README.md').write_text(
        '# PA / GraphSAGE reconstruction freeze\n\n'
        'This preserves the implementation and frozen reconstruction protocol, '
        'not the original implementation that produced the paper. '
        'Full training readiness is recorded separately from this source snapshot.\n\n'
        'The archive preserves source/configuration/binary bytes independently of future edits. '
        'It is a provenance archive, not a self-contained dataset or environment.\n\n'
        'Verify from the project root:\n\n```bash\n'
        'python ae/freeze.py verify frozen/' + snapshot.name + '\n'
        'python ae/freeze.py verify frozen/' + snapshot.name + ' --live\n```\n\n'
        'Read-only file modes prevent accidental edits; checksums detect drift. '
        'This is not WORM storage. New changes require a new snapshot ID.\n')
    for path in snapshot.iterdir():
        path.chmod(0o444)
    snapshot.chmod(0o555)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'verify'])
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--protocol', type=Path, help='Protocol to identify in a new snapshot')
    parser.add_argument('--role', default='paper_reconstruction_starting_implementation_and_protocol', help='New snapshot role; does not certify runtime acceptance')
    args = parser.parse_args()
    if args.action == 'create' and args.live:
        parser.error('--live is only for verify')
    result = create(args.snapshot,args.protocol,args.role) if args.action == 'create' else verify(args.snapshot, ROOT if args.live else None)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
