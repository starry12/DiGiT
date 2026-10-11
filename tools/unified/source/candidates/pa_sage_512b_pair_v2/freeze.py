"""Freeze the Python-only extent fix; keep the accepted build and small GPU tests."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Candidate already frozen')
    parent_sha=parent.verify();checks=read(OUT/'cpu_checks.json')
    require(checks['passed'],'CPU regressions failed')
    for rel,digest in checks['tested_sha256'].items():require(sha(ROOT/rel)==digest,'Tested source changed: '+rel)
    check_inputs(read(OUT/'inputs.json'))
    require(sha(OUT/'inputs.json')==sha(parent.OUT/'inputs.json'),'Input binding differs')
    require(sha(OUT/'gpu_checks.json')==sha(parent.OUT/'gpu_checks.json'),'GPU evidence differs')
    files={str(f.relative_to(ROOT)):sha(f) for f in HERE.iterdir() if f.is_file()}
    write(HERE/'manifest.json',dict(schema='pa-sage-512b-padding-fix-v2',parent_sha256=parent_sha,
        files=files,evidence_sha256={name:sha(OUT/name) for name in ('inputs.json','cpu_checks.json','gpu_checks.json')},
        created_unix=time.time()))
    print('Frozen padding repair:',verify(),flush=True)


if __name__=='__main__':main()
