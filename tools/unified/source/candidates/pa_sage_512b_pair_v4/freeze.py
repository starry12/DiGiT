"""Bind the bounded Python pipeline to the already accepted v3 native backend."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Already frozen')
    parent_sha=parent.verify();check_inputs(read(OUT/'inputs.json'))
    for name in ('cpu_checks.json','gpu_checks.json'):
        receipt=read(OUT/name);require(receipt['passed'],'Checks failed')
        for rel,digest in receipt['tested_sha256'].items():require(sha(ROOT/rel)==digest,'Tested source changed: '+rel)
    require(read(OUT/'gpu_checks.json')['binary_sha256']==sha(BINARY),'GPU backend differs')
    files={str(f.relative_to(ROOT)):sha(f) for f in HERE.iterdir() if f.is_file()}
    write(HERE/'manifest.json',dict(schema='pa-sage-bounded-accumulator-v4',files=files,
        parent_sha256=parent_sha,created_unix=time.time(),
        evidence_sha256={name:sha(OUT/name) for name in ('inputs.json','cpu_checks.json','gpu_checks.json')}))
    print('Frozen bounded accumulator:',verify(),flush=True)


if __name__=='__main__':main()
