"""Preserve v4 protocol/backend and bind this diagnostic independently."""
from candidates.pa_sage_512b_pair_v4.common import *
from candidates.pa_sage_512b_pair_v4 import common as baseline
from .profile import validate_profile

HERE = Path(__file__).resolve().parent
OUT = ROOT/'results/pa_sage_512b_sampling_profile_20260926_v1'
MODULE = 'candidates.pa_sage_512b_sampling_profile_v1'


def verify():
    m = read(HERE/'manifest.json')
    require(baseline.verify()==m['baseline_sha256'], 'Frozen v4 changed')
    for rel, digest in m['files'].items():
        require(sha(ROOT/rel)==digest, 'Profile source changed: '+rel)
    for name, digest in m['evidence_sha256'].items():
        require(sha(OUT/name)==digest, 'Profile preparation changed: '+name)
    require(sha(OUT/'inputs.json')==sha(baseline.OUT/'inputs.json'), 'Inputs changed')
    require(read(HERE/'protocol.json')==read(baseline.HERE/'protocol.json'), 'Training protocol changed')
    for path, digest in m['runtime_sha256'].items():
        require(sha(path)==digest, 'Profile runtime dependency changed: '+path)
    return sha(HERE/'manifest.json')


def validate_completion(report):
    baseline.validate_completion(report)
    validate_profile(report)
    return True


def full_schedule():
    # Each arm has off/host/host/off; reverse system order in the middle.
    return [('gids','off'), ('digit','off'), ('digit','host'), ('gids','host'),
            ('gids','host'), ('digit','host'), ('digit','off'), ('gids','off')]


def source_files():
    return sorted(f for f in HERE.iterdir() if f.is_file() and f.name!='manifest.json')
