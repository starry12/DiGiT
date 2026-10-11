"""Bind the continuation to unchanged native workers and completed pilot evidence."""
from pathlib import Path
from ae.common import read, write, sha, require, host

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'results/pa_sage_layout_shared_resume_20260925_v3'
OUT = ROOT / 'results/pa_sage_layout_continue_20260925_v4'
PLAN = SOURCE / 'layout_plan.json'
INDEX = SOURCE / 'protocols/index.json'
PY = '/home/embed/miniconda3/envs/gids/bin/python'
UNIT = 'digit-pa-layout-continue-20260925-v4'
NATIVE = 'candidates.pa_sage_layout_shared_resume_v3'
NATIVE_SHA = 'fe9c705882a57da6c25b7978076b598cd984fcba674b8fffb80e8dee84eb2bd3'
PILOTS = [('real', 'calibration/real_first'), ('shared', 'native/g2_r20'),
          ('shared', 'calibration/shared_repeat'), ('real', 'calibration/real_second')]


def verify():
    from candidates.pa_sage_layout_shared_resume_v3.common import verify as native_verify
    require(native_verify() == NATIVE_SHA, 'Frozen native v3 changed')
    m = read(HERE / 'manifest.json')
    require(m['native_candidate_sha256'] == NATIVE_SHA, 'Wrong native lineage')
    for name, digest in m['files'].items():
        require(sha(HERE / name) == digest, 'Continuation source changed: ' + name)
    for name, digest in m['evidence'].items():
        require(sha(ROOT / name) == digest, 'Completed evidence/configuration changed: ' + name)
    return sha(HERE / 'manifest.json')
