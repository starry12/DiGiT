"""Freeze only after CPU checks; all parent data and native code stay unchanged."""
from .common import *


def main():
    from candidates.ig_sage_stage_profile_v1.common import verify as parent
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
    gpu=read(OUT/'gpu_checks.json');require(gpu['passed'] and gpu['all_native_blocks_logits_gradients_loss_parameters_adam_exact'],'GPU equivalence required')
    write(HERE/'manifest.json',dict(schema='digit-ig-host-opt-candidate-v1',
        parent_candidate_sha256=PARENT_SHA,files=files,protocol_sha256=sha(P),
        optimization_protocol_sha256=sha(HERE/'optimization_protocol.json'),cpu_checks_sha256=sha(OUT/'cpu_checks.json'),
        gpu_checks_sha256=sha(OUT/'gpu_checks.json'),
        native_backend_changed=False,formal_added_cuda_synchronization=False,separate_probe_cuda_events=True,
        scope='DiGiT IG/SAGE: combined native smoke, four fresh 20+300 variants, separate 8-batch native sampler event probe',
        native_acceptance='pending large-graph smoke and full bounded comparison'))
    print(json.dumps(dict(candidate_sha256=verify(),files=len(files)),indent=2))


if __name__=='__main__':main()
