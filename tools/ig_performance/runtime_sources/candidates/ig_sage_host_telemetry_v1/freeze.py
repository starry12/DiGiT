"""Freeze orchestration changes with existing verified original native binaries."""
from .common import *


def main():
    from candidates.ig_sage_affinity_pair_v1.common import verify as parent
    require(not (HERE/'manifest.json').exists(),'Already frozen; create a new candidate')
    require(parent()==PARENT_SHA,'Accepted parent changed')
    checks=read(OUT/'cpu_checks.json');require(checks['passed'],'CPU checks required')
    for path,expected in checks['tested_sha256'].items():
        require(sha(ROOT/path)==expected,'Tested source changed: '+path)
    for name,expected in checks['evidence_sha256'].items():
        require(sha(OUT/name)==expected,'Test evidence changed: '+name)
    require(sha(HERE/'runtime/IGPerfNative.so')==sha(ROOT/'candidates/ig_sage_host_opt_v1/runtime/IGPerfNative.so'),'Original native binary changed')
    require(read(OUT/'telemetry_probe.json')['passed'] and read(OUT/'inherited_smokes.json')['passed'],'Probe and inherited smoke required')
    probe_files={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'telemetry_probe_raw').rglob('*')) if p.is_file()}
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write(HERE/'manifest.json',dict(schema='digit-ig-host-telemetry-candidate-v1',parent_candidate_sha256=PARENT_SHA,
        files=files,protocol_sha256=sha(P),optimization_protocol_sha256=sha(HERE/'optimization_protocol.json'),
        cpu_checks_sha256=sha(OUT/'cpu_checks.json'),native_backend_changed=False,
        inherited_smokes_sha256=sha(OUT/'inherited_smokes.json'),telemetry_probe_sha256=sha(OUT/'telemetry_probe.json'),
        telemetry_probe_files=probe_files,
        formal_added_cuda_synchronization=False,profile_mode='host',separate_probe=False,
        scope='Original GIDS default versus original DiGiT CPU2; reused native smokes then host-profile ABBA 20+300 batches with scheduling/clock telemetry',
        native_acceptance='inherited correctness accepted; fresh diagnostic training pending'))
    print(json.dumps(dict(candidate_sha256=verify(),files=len(files)),indent=2))


if __name__=='__main__':main()
