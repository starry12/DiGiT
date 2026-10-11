"""Bind the independent source, build, CPU checks and read-only inputs."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(), 'Candidate already frozen')
    build=read(HERE/'runtime/build_receipt.json')
    native=native_sources()
    require(build['native_source_sha256']==object_sha(native) and build['binary_sha256']==sha(BINARY),
            'Native build is stale')
    cpu=read(OUT/'cpu_checks.json')
    require(cpu['passed'], 'CPU tests failed')
    for path,digest in cpu['tested_sha256'].items():
        require(sha(HERE/path)==digest,'Tested source changed: '+path)
    require(read(OUT/'abi_check.json')['binary_sha256']==sha(BINARY),'ABI check is stale')
    check_inputs(read(OUT/'inputs.json'))
    files={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob('*')
           if p.is_file() and p.suffix in ('.py','.json','.h','.cu','.cuh','.md') and
           '__pycache__' not in p.parts}
    dependencies={k:v for k,v in native.items() if not k.startswith(str(HERE.relative_to(ROOT))+'/')}
    for path in ['ae/common.py','ae/pa_sage/common.py','digit_paths.py','evaluation/models.py',
                 'configs/papers/protocol.json','configs/external_paths.json','configs/ssd_payloads.json',
                 'configs/device.json','ae/native/identity/identify-module',
                 str((OUT/'cpu_checks.json').relative_to(ROOT)),str((OUT/'abi_check.json').relative_to(ROOT))]:
        dependencies[path]=sha(ROOT/path)
    write(HERE/'manifest.json',dict(schema='pa-legacy-align-frozen-v1',files=files,
        dependencies=dependencies,binary_sha256=sha(BINARY),inputs_sha256=sha(OUT/'inputs.json'),
        native_source_sha256=build['native_source_sha256'],created_unix=time.time()))
    print('Frozen candidate SHA256:',verify(),flush=True)


if __name__=='__main__':
    main()
