"""Freeze the Python-only adaptation and bind the previously accepted backend."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(), 'Candidate already frozen')
    parent = legacy.verify()
    check_inputs(read(OUT/'inputs.json'))
    require(sha(OUT/'inputs.json') == sha(legacy.OUT/'inputs.json'), 'Expected exact inherited input binding')
    checks = read(OUT/'cpu_checks.json')
    require(checks['passed'], 'CPU checks failed')
    for rel, digest in checks['tested_sha256'].items():
        require(sha(ROOT/rel) == digest, 'Tested source changed: '+rel)
    files = {str(f.relative_to(ROOT)):sha(f) for f in sorted(HERE.rglob('*'))
             if f.is_file() and f.suffix in ('.py','.json','.md') and '__pycache__' not in f.parts}
    write(HERE/'manifest.json', dict(schema='pa-sage-cpu-affinity-frozen-v1',files=files,
        parent_manifest_sha256=parent,binary_sha256=sha(BINARY),inputs_sha256=sha(OUT/'inputs.json'),
        cpu_checks_sha256=sha(OUT/'cpu_checks.json'),created_unix=time.time()))
    print('Frozen CPU affinity candidate:',verify(),flush=True)


if __name__ == '__main__':
    main()
