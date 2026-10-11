"""Independent ablation contract over the immutable accepted native runtime."""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'candidates/pa_sage_bidir_native_v2'
P = HERE / 'protocol.json'
sys.path.insert(0, str(ROOT))
from ae.common import sha, write, require, host
from ae.pa_sage.common import source_config, splits

def read(path):
    return json.loads(Path(path).read_text())

def cfg():
    return read(P)

def setup():
    from candidates.pa_sage_bidir_native_v2.common import setup as parent_setup
    parent_setup()

def progress(output, stage, **kw):
    value = dict(stage=stage, pid=os.getpid(), time_unix=time.time(), **kw)
    write(Path(output) / 'progress.json', value)
    print(json.dumps(value), flush=True)

def verify():
    from candidates.pa_sage_bidir_native_v2.common import verify as parent_verify
    manifest = read(HERE / 'manifest.json')
    require(parent_verify() == manifest['parent_candidate_sha256'], 'Parent runtime changed')
    for name, digest in manifest['files'].items():
        require(sha(HERE / name) == digest, 'Ablation candidate changed: ' + name)
    for name,digest in manifest['external_files'].items():
        require(sha(ROOT/name)==digest,'Ablation launcher changed: '+name)
    return sha(HERE / 'manifest.json')
