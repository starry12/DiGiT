"""Bind a new experiment before checks; never rewrite an existing manifest."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Preserve the existing source manifest')
    setup()
    receipt=read(HERE/'runtime/build_receipt.json')
    sources=native_sources()
    require(receipt['passed'] and receipt['native_sources']==sources,'Stale native build')
    require(receipt['binary_sha256']==sha(BINARY),'Build binary hash differs')
    files=list(HERE.glob('*.py'))+[HERE/'protocol.json',HERE/'README.md',HERE/'identity_registry.json',
        HERE/'runtime/BAM_Feature_Store/__init__.py',HERE/'runtime/build_receipt.json']
    m=dict(reference_sha256=reference.verify(),files={str(p.relative_to(HERE)):sha(p) for p in files},
        native_sources=sources,binary_sha256=sha(BINARY),inputs_sha256=sha(OUT/'inputs.json'))
    write(HERE/'manifest.json',m)
    print('Frozen candidate '+verify(),flush=True)


if __name__=='__main__':main()
