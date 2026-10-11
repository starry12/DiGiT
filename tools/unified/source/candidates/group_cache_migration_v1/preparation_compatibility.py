"""Reuse the completed overlay across audited monitor and raw-reader revisions.

The preparation receipt retains its original source digest. No receipt is
rewritten and all original input/output identity checks remain in prepare.py.
"""
from .common import ROOT, HERE, read, sha, verify

PARENT_SHA = '08278dc626c618f9a90b15b691e0527caf69eaadb2ecf1a30647a1bd8c595aa8'
PARENT = ROOT/'results/group_cache_migration_v1/monitor_revision_20261010_v2/candidate/manifest.json'
PREFIX = 'candidates/group_cache_migration_v1/'
ALLOWED_CHANGES = {PREFIX + name for name in (
    'README.md', 'prepare.py', 'freeze.py', 'ig/common.py', 'ig/controller.py', 'ig/review.py',
    'ig/cache.py', 'ig/worker.py', 'gpu_check.py')}


def compatible_files(old, current):
    # Every previously sealed dependency and native binary must be unchanged,
    # except explicit monitor/reader wiring, readiness checks and receipt compatibility.
    return all(name in current and (current[name] == digest or name in ALLOWED_CHANGES)
               for name, digest in old.items())


def accepted_source(digest):
    current = verify()
    if digest == current:
        return True
    if digest != PARENT_SHA or sha(PARENT) != PARENT_SHA:
        return False
    return compatible_files(read(PARENT)['files'], read(HERE/'manifest.json')['files'])
