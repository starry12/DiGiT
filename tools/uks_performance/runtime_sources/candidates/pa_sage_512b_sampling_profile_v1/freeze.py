"""Bind tested profile code, unchanged v4, DGL code and input identities."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Already frozen')
    base=baseline.verify();check_inputs(read(OUT/'inputs.json'))
    for name in ('cpu_checks.json','gpu_checks.json'):
        result=read(OUT/name);require(result['passed'],'Checks failed')
        for rel,digest in result['tested_sha256'].items():
            require(sha(ROOT/rel)==digest,'Tested code changed: '+rel)
    setup()
    import dgl.dataloading.neighbor_sampler as ns
    import dgl.dataloading.base as bs
    runtime=[ns.__file__,bs.__file__]
    write(HERE/'manifest.json',dict(schema='pa-sage-512b-sampling-profile-v1',
        files={str(f.relative_to(ROOT)):sha(f) for f in source_files()},baseline_sha256=base,
        runtime_sha256={f:sha(f) for f in runtime},created_unix=time.time(),
        evidence_sha256={name:sha(OUT/name) for name in ('inputs.json','cpu_checks.json','gpu_checks.json')}))
    print('Frozen sampling diagnostic:',verify(),flush=True)


if __name__=='__main__':main()
