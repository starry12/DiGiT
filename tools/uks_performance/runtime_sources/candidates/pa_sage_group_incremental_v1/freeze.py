"""Bind a tested standalone sampler to the already accepted asynchronous I/O."""
from .common import *


def main():
    require(not (HERE / 'manifest.json').exists(), 'Already frozen; use a new candidate')
    reference_sha = reference.verify()
    check_inputs(read(OUT / 'inputs.json'))
    names = ('cpu_checks.json', 'gpu_checks.json', 'native_checks.json')
    for name in names:
        r = read(OUT / name)
        require(r['passed'], 'Failed prerequisite: ' + name)
        for rel, digest in r.get('tested_sha256', {}).items():
            require(sha(ROOT / rel) == digest, 'Tested source changed: ' + rel)
    gpu = read(OUT / 'gpu_checks.json')
    require(gpu['reference_sha256']==reference_sha,'GPU reference changed')
    require(gpu['binary_sha256'] == sha(BINARY), 'GPU evidence I/O binary differs')
    require(gpu['sampler_binary_sha256'] == sha(SAMPLER_BINARY), 'GPU evidence group binary differs')
    native = read(OUT / 'native_checks.json')
    require(native['binary_sha256'] == sha(SAMPLER_BINARY), 'Native evidence binary differs')
    receipt = read(HERE / 'runtime/build_receipt.json')
    require(receipt['passed'] and receipt['binary_sha256'] == sha(SAMPLER_BINARY), 'Build binding differs')
    for rel, digest in receipt['source_sha256'].items():
        require(sha(ROOT / rel) == digest, 'Build source changed: ' + rel)
    require(all(receipt['preserved_kernels'][k] for k in
        ('group_kernel_byte_identical', 'legacy_eid_kernel_byte_identical')), 'Original kernels changed')
    write(HERE / 'manifest.json', dict(schema='pa-sage-group-incremental-v1', reference_sha256=reference_sha,
        created_unix=time.time(), files={str(p.relative_to(ROOT)): sha(p) for p in source_files()},
        binary_sha256=sha(BINARY), sampler_binary_sha256=sha(SAMPLER_BINARY),
        evidence_sha256={name: sha(OUT / name) for name in ('inputs.json',) + names}))
    print('Frozen incremental group candidate:', verify(), flush=True)


if __name__ == '__main__':
    main()
