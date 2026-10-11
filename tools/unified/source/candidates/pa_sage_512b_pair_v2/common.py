"""Padding validation repair; reuse the frozen 512-B runtime and input binding."""
from candidates.pa_sage_512b_pair_v1.common import *
from candidates.pa_sage_512b_pair_v1 import common as parent

HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/pa_sage_512b_pair_20260926_v2'
MODULE='candidates.pa_sage_512b_pair_v2'


def verify():
    m=read(HERE/'manifest.json')
    require(parent.verify()==m['parent_sha256'],'Frozen parent changed')
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'Repair changed: '+rel)
    for name in ('inputs.json','cpu_checks.json','gpu_checks.json'):
        require(sha(OUT/name)==m['evidence_sha256'][name],'Evidence changed: '+name)
    require(sha(OUT/'inputs.json')==sha(parent.OUT/'inputs.json'),'Different input binding')
    require(sha(OUT/'gpu_checks.json')==sha(parent.OUT/'gpu_checks.json'),'Inherited GPU checks changed')
    require(sha(HERE/'protocol.json')==sha(parent.HERE/'protocol.json'),'Experiment protocol changed')
    return sha(HERE/'manifest.json')


def validate_completion(report):
    parent.validate_completion(report)
    from .payload import validate_payload
    expected=validate_payload(read(OUT/'inputs.json'),read(HERE/'protocol.json'),report['arm'])
    require(report['payload_extent']==expected,'Missing or altered payload extent audit')
    return True


def compare(gids,digit):
    for r in (gids,digit):validate_completion(r)
    return parent.compare(gids,digit)
