"""Freeze only after CPU, simulated pipeline and native async state-machine checks."""
from .common import *
from .native_sources import native_sources,object_sha as native_object_sha


def main():
    require(not (HERE/'manifest.json').exists(),'Already frozen; use a new candidate')
    reference_sha=reference.verify();check_inputs(read(OUT/'inputs.json'))
    names=('cpu_checks.json','gpu_checks.json','native_probe.json')
    for name in names:
        r=read(OUT/name);require(r['passed'],'Failed prerequisite: '+name)
        for rel,digest in r.get('tested_sha256',{}).items():require(sha(ROOT/rel)==digest,'Tested source changed: '+rel)
    require(read(OUT/'gpu_checks.json')['binary_sha256']==sha(BINARY),'GPU evidence binary differs')
    probe=read(OUT/'native_probe.json')
    require(probe['binary_sha256']==sha(BINARY),'Native probe binary differs')
    require(probe['native_source_sha256']==native_object_sha(native_sources()),'Native probe source differs')
    require(probe['probe_python_sha256']==sha(HERE/'native_probe.py'),'Native probe Python changed')
    require(read(HERE/'runtime/build_receipt.json')['native_source_sha256']==native_object_sha(native_sources()),
        'Build source binding differs')
    write(HERE/'manifest.json',dict(schema='pa-sage-512b-overlap-v1',reference_sha256=reference_sha,
        created_unix=time.time(),files={str(p.relative_to(ROOT)):sha(p) for p in source_files()},
        native_sources=native_sources(),binary_sha256=sha(BINARY),
        evidence_sha256={name:sha(OUT/name) for name in ('inputs.json',)+names}))
    print('Frozen read/compute overlap candidate:',verify(),flush=True)


if __name__=='__main__':main()
