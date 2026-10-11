"""Freeze the tested constructor, native victim scan and unchanged protocol."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Candidate already frozen')
    parent_sha=parent.verify();check_inputs(read(OUT/'inputs.json'))
    checks=[read(OUT/name) for name in ('cpu_checks.json','gpu_checks.json')]
    for receipt in checks:
        require(receipt['passed'],'Checks failed')
        for rel,digest in receipt['tested_sha256'].items():require(sha(ROOT/rel)==digest,'Tested source changed: '+rel)
    build=read(HERE/'runtime/build_receipt.json')
    require(build['native_source_sha256']==object_sha(native_sources()),'Build source differs')
    require(build['binary_sha256']==sha(BINARY)==checks[1]['binary_sha256'],'Build/GPU identity mismatch')
    files={str(f.relative_to(ROOT)):sha(f) for f in HERE.rglob('*') if f.is_file()
           and f.suffix in ('.py','.md','.json') and '__pycache__' not in f.parts}
    files.update(native_sources())
    write(HERE/'manifest.json',dict(schema='pa-sage-512b-fifo-window-v3',files=files,
        parent_sha256=parent_sha,binary_sha256=sha(BINARY),
        evidence_sha256={name:sha(OUT/name) for name in ('inputs.json','cpu_checks.json','gpu_checks.json')},
        created_unix=time.time()))
    print('Frozen FIFO/window candidate:',verify(),flush=True)


if __name__=='__main__':main()
