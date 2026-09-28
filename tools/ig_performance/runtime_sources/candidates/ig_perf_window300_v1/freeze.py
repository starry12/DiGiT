"""Freeze this protocol-only candidate once; never re-sign old versions."""
from .common import *

def main():
    from candidates.ig_perf_v5.common import verify as parent
    require(not (HERE/'manifest.json').exists(),'Already frozen; create a new candidate')
    require(parent()==PARENT_SHA,'Accepted parent changed')
    checks=read(OUT/'cpu_checks.json')
    require(checks['passed'],'CPU checks required')
    for path,digest_ in checks['tested_sha256'].items():
        require(sha(ROOT/path)==digest_,'Tested source changed: '+path)
    require(sha(HERE/'runtime/IGPerfNative.so')==sha(ROOT/'candidates/ig_perf_v5/runtime/IGPerfNative.so'),
            'Native binary must remain identical')
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
    write(HERE/'manifest.json',dict(schema='digit-ig-window300-candidate-v1',
        parent_candidate_sha256=PARENT_SHA,files=files,protocol_sha256=sha(P),
        cpu_checks_sha256=sha(OUT/'cpu_checks.json'),
        scope='20 warmup plus 3x100 measured batches; no validation/test/accuracy/epoch extrapolation',
        native_backend_changed=False,native_acceptance='fresh paired smoke and 300-batch comparison pending'))
    print(json.dumps(dict(candidate_sha256=verify(),files=len(files)),indent=2))

if __name__=='__main__':main()
