"""Bind this adaptation, its tested dependencies and the independently built backend."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Candidate already frozen')
    parent=parents();check_inputs(read(OUT/'inputs.json'))
    checks=read(OUT/'cpu_checks.json');gpu=read(OUT/'gpu_checks.json')
    require(checks['passed'] and gpu['passed'],'Checks did not pass')
    for receipt in (checks,gpu):
        for rel,digest in receipt['tested_sha256'].items():
            require(sha(ROOT/rel)==digest,'Tested source changed: '+rel)
    build=read(HERE/'runtime/build_receipt.json')
    require(build['native_source_sha256']==object_sha(native_sources()) and
            build['binary_sha256']==sha(BINARY)==gpu['binary_sha256'],'Build or GPU test identity mismatch')
    files={str(f.relative_to(ROOT)):sha(f) for f in sorted(HERE.rglob('*'))
           if f.is_file() and f.suffix in ('.py','.json','.md') and '__pycache__' not in f.parts}
    files.update(native_sources())
    write(HERE/'manifest.json',dict(schema='pa-sage-512b-pair-frozen-v1',files=files,
        parents=parent,binary_sha256=sha(BINARY),inputs_sha256=sha(OUT/'inputs.json'),
        cpu_checks_sha256=sha(OUT/'cpu_checks.json'),gpu_checks_sha256=sha(OUT/'gpu_checks.json'),
        created_unix=time.time()))
    print('Frozen independent candidate:',verify(),flush=True)


if __name__=='__main__':main()
