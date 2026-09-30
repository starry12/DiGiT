"""Immutable wrapper around the previous non-perf control worker."""
from pathlib import Path
from candidates.ig_sage_host_telemetry_v1.common import (
    ROOT, environment, read, write, sha, require, host, check_inputs, setup)
from candidates.ig_sage_host_telemetry_v1.common import verify as parent_verify

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/ig_sage_pair_max5_20260928_v1'
PARENT_SHA = 'f3651dec530fe91f18798c4cd9735f4200f07e4c7400ae801182760b4bb91827'
PY = '/home/embed/miniconda3/envs/gids/bin/python'


def verify():
    require(parent_verify() == PARENT_SHA, 'Original native worker changed')
    manifest = read(HERE / 'manifest.json')
    require(manifest['parent_sha256'] == PARENT_SHA, 'Wrong native parent')
    for name, expected in manifest['files'].items():
        require(sha(HERE / name) == expected, 'Five-round source changed: ' + name)
    return sha(HERE / 'manifest.json')
