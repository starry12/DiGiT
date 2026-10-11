"""Permit only authenticated filesystem renumbering; retain every other field."""
from pathlib import Path


def filesystem_uuid(path):
    dev=Path(path).stat().st_dev
    matches=[]
    for link in Path('/dev/disk/by-uuid').iterdir():
        try:
            if link.stat().st_rdev==dev:matches.append(link.name)
        except FileNotFoundError:continue
    if len(matches)!=1:raise RuntimeError('Filesystem UUID unavailable/ambiguous: '+str(path))
    return matches[0]


def verify_input(path,expected,registration,stat_identity):
    path=Path(path)
    if path.is_symlink() or '..' in path.parts:raise RuntimeError('Unsafe input path: '+str(path))
    if expected!=registration['identity']:raise RuntimeError('Historical input identity changed: '+str(path))
    before=stat_identity(path)
    if set(before)!=set(expected) or any(before[k]!=expected[k] for k in expected if k!='device'):
        raise RuntimeError('Input changed beyond device renumbering: '+str(path))
    if filesystem_uuid(path)!=registration['filesystem_uuid']:raise RuntimeError('Filesystem UUID changed: '+str(path))
    if stat_identity(path)!=before:raise RuntimeError('Input changed during verification: '+str(path))
    return before
