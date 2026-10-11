"""Freeze only after native CUDA capture/autograd and scheduling parity pass."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(), 'Already frozen; use a new candidate')
    reference_sha = reference.verify(); check_inputs(read(OUT/'inputs.json'))
    names = ('cpu_checks.json', 'gpu_checks.json', 'trace_probe.json')
    for name in names:
        r = read(OUT/name); require(r['passed'], 'Failed prerequisite: '+name)
        for rel, digest in r.get('tested_sha256', {}).items():
            require(sha(ROOT/rel) == digest, 'Tested source changed: '+rel)
    gpu = read(OUT/'gpu_checks.json')
    require(gpu['binary_sha256'] == sha(BINARY), 'GPU evidence binary differs')
    require(gpu['reference_sha256'] == reference_sha, 'Reference changed since GPU checks')
    require(gpu['environment_files'] == environment_files(), 'CUDA Graph implementation changed')
    write(HERE/'manifest.json', dict(schema='pa-sage-dense-graph-v1', reference_sha256=reference_sha,
        created_unix=time.time(), files={str(p.relative_to(ROOT)):sha(p) for p in source_files()},
        binary_sha256=sha(BINARY), environment_files=environment_files(),
        evidence_sha256={name:sha(OUT/name) for name in ('inputs.json',)+names}))
    print('Frozen dense CUDA Graph candidate:',verify(),flush=True)


if __name__ == '__main__':main()
