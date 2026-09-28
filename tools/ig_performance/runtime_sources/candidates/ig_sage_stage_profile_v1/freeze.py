"""Freeze only after CPU checks; all parent data and native code stay unchanged."""
from .common import *


def main():
    from candidates.ig_perf_window300_v1.common import verify as parent
    require(not (HERE/'manifest.json').exists(),'Already frozen; create a new candidate')
    require(parent()==PARENT_SHA,'Accepted parent changed')
    checks=read(OUT/'cpu_checks.json')
    require(checks['passed'],'CPU checks required')
    for path,digest_ in checks['tested_sha256'].items():
        require(sha(ROOT/path)==digest_,'Tested source changed: '+path)
    for name,digest_ in checks['evidence_sha256'].items():
        require(sha(OUT/name)==digest_,'Test evidence changed: '+name)
    require(sha(HERE/'runtime/IGPerfNative.so')==sha(ROOT/'candidates/ig_perf_window300_v1/runtime/IGPerfNative.so'),'Native binary changed')
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write(HERE/'manifest.json',dict(schema='digit-ig-symmetric-stage-candidate-v1',
        parent_candidate_sha256=PARENT_SHA,files=files,protocol_sha256=sha(P),
        profile_protocol_sha256=sha(HERE/'profile_protocol.json'),cpu_checks_sha256=sha(OUT/'cpu_checks.json'),
        native_backend_changed=False,added_cuda_synchronization=False,
        scope='IG/SAGE: paired host smoke followed by fresh off/host workers per arm; 20 warmup plus 300 timed batches each',
        native_acceptance='pending fresh paired smoke and off/host comparison'))
    print(json.dumps(dict(candidate_sha256=verify(),files=len(files)),indent=2))


if __name__=='__main__':main()
