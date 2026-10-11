"""Bind the independent tested normalization module and unchanged reference."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Already frozen; use a new candidate')
    reference_sha=reference.verify();check_inputs(read(OUT/'inputs.json'))
    names=('cpu_checks.json','gpu_checks.json','native_checks.json')
    for name in names:
        r=read(OUT/name);require(r['passed'],'Failed prerequisite: '+name)
        for rel,digest in r.get('tested_sha256',{}).items():require(sha(ROOT/rel)==digest,'Tested source changed: '+rel)
    gpu=read(OUT/'gpu_checks.json')
    require(gpu['reference_sha256']==reference_sha and gpu['model_and_adam_bit_exact'],'GPU reference/model changed')
    for path,key in ((BINARY,'binary_sha256'),(SAMPLER_BINARY,'sampler_binary_sha256'),(NORMALIZATION_BINARY,'normalization_binary_sha256')):
        require(gpu[key]==sha(path),'GPU binary changed: '+key)
    require(gpu['environment_files']==environment_files(),'GPU environment changed')
    require(read(OUT/'native_checks.json')['binary_sha256']==sha(NORMALIZATION_BINARY),'Native evidence binary changed')
    receipt=read(HERE/'runtime/build_receipt.json')
    require(receipt['passed'] and receipt['binary_sha256']==sha(NORMALIZATION_BINARY),'Build binding differs')
    for rel,digest in receipt['source_sha256'].items():require(sha(ROOT/rel)==digest,'Built source changed: '+rel)
    write(HERE/'manifest.json',dict(schema='pa-sage-mean-fused-v1',reference_sha256=reference_sha,
        created_unix=time.time(),files={str(p.relative_to(ROOT)):sha(p) for p in source_files()},
        binary_sha256=sha(BINARY),sampler_binary_sha256=sha(SAMPLER_BINARY),normalization_binary_sha256=sha(NORMALIZATION_BINARY),
        environment_files=environment_files(),evidence_sha256={n:sha(OUT/n) for n in ('inputs.json',)+names}))
    print('Frozen mean normalization candidate:',verify(),flush=True)


if __name__=='__main__':main()
