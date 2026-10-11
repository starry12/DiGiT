"""Reuse only the exact pre-label-fix preparation/profile, without relabeling it."""
PREVIOUS = 'bdc5b386da0fff8a7e3b367018afef3963bad95ce765ee51c1811a16c5d99c8c'
CHANGED = {'common.py', 'engine.py', 'controller.py', 'freeze.py', 'tests.py'}
ADDED = {'labels.py', 'provenance.py', 'prior_manifest.json'}


def check_transition(previous, current):
    if previous['dependencies'] != current['dependencies']:
        raise RuntimeError('Preparation/profile dependency changed')
    old, new = previous['files'], current['files']
    if set(old) - set(new) or set(new) - set(old) != ADDED:
        raise RuntimeError('Unexpected source set for label-only repair')
    changed = {name for name in old if old[name] != new[name]}
    if not changed <= CHANGED:
        raise RuntimeError('Preparation/profile source changed: ' + str(sorted(changed - CHANGED)))


def check_prepared_source(source):
    from .common import HERE, read, sha, verify, require
    current = verify()
    if source == current:
        return
    require(source == PREVIOUS and sha(HERE/'prior_manifest.json') == PREVIOUS,
            'Unknown preparation/profile provenance')
    check_transition(read(HERE/'prior_manifest.json'), read(HERE/'manifest.json'))
